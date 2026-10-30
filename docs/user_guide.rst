.. _user_guide:

User Guide
==========

This guide provides detailed instructions on using the core features of the HEonGPU library. It covers fundamental concepts, parameter selection, data management, and includes tutorials for common use cases based on the BFV and CKKS homomorphic encryption schemes.

Core Library Concepts
---------------------

The HEonGPU API is designed to be stable, intuitive, and to abstract away the underlying CUDA complexity. Developers primarily interact with a set of high-level C++ classes that manage the entire lifecycle of homomorphic encryption operations.

* ``HEContext``: This is the central class that manages the cryptographic environment. It is initialized with a set of encryption parameters and, once generated, the context becomes immutable. It validates parameters, pre-computes necessary values on the GPU (such as tables for the Number Theoretic Transform), and governs all subsequent operations.
* ``HEKeyGenerator``: Responsible for creating all necessary cryptographic keys from a given context. This includes the public/secret key pair, as well as **relinearization keys** and **Galois keys** required for specific homomorphic operations like multiplication and rotations.
* ``HEEncoder``: Translates user data (e.g., integers, real numbers) into the library's ``Plaintext`` format, which is the representation required for encryption. The encoding method is scheme-specific.
* ``HEEncryptor`` / ``HEDecryptor``: These classes handle the core encryption and decryption operations, using the public and secret keys, respectively.
* ``HEArithmeticOperator``: The workhorse of the library. This class performs all homomorphic evaluations (e.g., addition, multiplication, relinearization, rotation) on ciphertexts. All operations within this class are executed entirely on the GPU for maximum performance.
* ``ExecutionOptions``: A utility struct that gives the user fine-grained control over data locality (i.e., whether data resides on the host CPU or the device GPU) during and after an operation.

Most functions write their result into an object passed by the caller, which can be reused. The main ones also have versions that return the result, which give the same results:

.. code-block:: cpp

    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    auto plaintext = encoder.encode(message, scale);
    auto x = encryptor.encrypt(plaintext);
    auto sum = operators.add(x, x);
    auto decrypted = decryptor.decrypt(sum);
    auto values = encoder.decode(decrypted);

The inputs are taken by ``const`` reference, so the result of one call can be passed directly to the next, as in ``decryptor.decrypt(operators.add(x, x))``.

For vectors packed into the slots, ``sum_slots(ciphertext, galois_key)`` puts the sum of all slots into every slot, and ``inner_product(x, y, relin_key, galois_key)`` the inner product of two ciphertexts (or of a ciphertext and a plaintext). They need the Galois keys for the steps 1, 2, 4, ... up to half the slot count, as ``Galoiskey(context, log2(slot count) - 1)`` has; for CKKS, the default ``Galoiskey`` also works, with more rotations.

``rotate_hoisted(ciphertext, galois_key, shifts)`` (``rotate_rows_hoisted`` for BFV) returns the rotations of one ciphertext by several shifts, with the results of ``rotate`` (``rotate_rows``). Rotations by shifts that the Galois key has a key for share the first part of key switching, the decomposition of the ciphertext, so that 8 rotations take about a third less time than 8 single rotations. This helps where one ciphertext is rotated by many steps, as in matrix-vector products.

CKKS ciphertexts can be put through polynomials, and through functions approximated by them. The result has the scale of the input:

.. code-block:: cpp

    // 0.5 + 0.25 x - 0.02 x^3, for slots x with |x| <= 1.
    auto p = operators.evaluate_polynomial(x, {0.5, 0.25, 0.0, -0.02}, relin_key);

    // The sigmoid on [-8, 8], approximated by a Chebyshev interpolant of degree 31.
    auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    auto s = operators.evaluate_chebyshev(x, sigmoid, -8.0, 8.0, 31, relin_key);

A polynomial of degree :math:`d` uses :math:`\lceil \log_2(d + 1) \rceil` levels, and a Chebyshev series on an interval other than :math:`[-1, 1]` one more; ``HEArithmeticOperator<Scheme::CKKS>::polynomial_depth`` returns this number. The slots must lie in the interval of the approximation. ``chebyshev_coefficients(function, a, b, degree)`` returns the coefficients of the interpolant, so that its error can be checked on the CPU, and ``evaluate_chebyshev(x, coefficients, a, b, relin_key)`` evaluates given coefficients.

CKKS bootstrapping needs a context, keys and a configuration that fit together. ``CKKSBootstrappingPreset::regular()``, ``slim()``, ``bit()`` and ``gate()`` hold the parameters of the bootstrapping examples (:math:`N = 2^{16}`, 128-bit security), and ``HEKeyGenerator::bootstrapping_keys`` generates all keys of a bootstrapping in one call:

.. code-block:: cpp

    auto preset = heongpu::CKKSBootstrappingPreset::regular();
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(heongpu::sec_level_type::sec128);
    preset.apply(context);
    context->generate();

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, preset.secret_key_hamming_weight);
    keygen.generate_secret_key(secret_key);
    // ... public key, encoder, encryptor, decryptor and operators

    operators.generate_bootstrapping_params(preset);
    auto boot_keys = keygen.bootstrapping_keys(secret_key, operators.bootstrapping_key_indices(),
                                               preset.sparse_secret_key_hamming_weight);
    auto refreshed = operators.regular_bootstrapping(ciphertext, boot_keys);

Bootstrapping brings a ciphertext above the level it starts from down to that level. ``preset.levels()``, or ``bootstrapping_levels(config, type)`` for another configuration, returns the levels it uses: ``input_level`` and ``output_level``, and ``usable_levels()`` for the computation between two bootstrappings (9 for regular and slim, 8 for bit and 7 for gate bootstrapping with the presets). ``regular(levels)``, ``slim(levels)``, ``bit(levels)`` and ``gate(levels)`` leave ``levels`` levels instead, each a 40-bit prime, and move the configuration to them; at 128-bit security at most 14 fit with regular and slim, 12 with bit and 11 with gate bootstrapping.

Parameter Selection Guide
-------------------------

Choosing the right encryption parameters is crucial for balancing security, performance, and the computational depth of the circuits you can evaluate. In HEonGPU, all parameters are set at the ``HEContext`` level before key generation, either by hand or by the parameter generator ``HEParams``.

* **Security Level**: The library is designed and tested for a 128-bit security level, which is the standard for most modern applications.
* **Polynomial Modulus Degree (N)**: This parameter is the primary driver of security and performance. Larger values of :math:`N` provide more security and a larger "noise budget" (allowing for more computations), but at a significant performance cost. The library is optimized for power-of-two values for :math:`N` in the range of :math:`[2^{12}, 2^{16}]`.
* **Coefficient Modulus (q)**: This is a chain of prime numbers whose product forms the ciphertext modulus. The total bit-size of :math:`q` determines the computational depth—a larger modulus allows for more sequential multiplications. The library's use of the Residue Number System (RNS) makes operations with these prime chains highly suitable for parallelization on the GPU.
* **Plaintext Modulus (t)**: For the BFV scheme, this parameter defines the size of the integer message space (i.e., computations are performed modulo :math:`t`).

``HEParams`` chooses :math:`N` and the primes from what the computation needs: the multiplicative depth, the security level, and the scale for CKKS or the plaintext modulus for BFV. It takes the smallest :math:`N` whose security limit fits them; the parameters it returns never exceed the limit.

.. code-block:: cpp

    heongpu::HEParams<heongpu::Scheme::CKKS> params;
    params.multiplicative_depth = 10;
    params.scale_bit_size = 40;  // the default
    auto parameters = params.generate();
    std::cout << parameters << std::endl;
    // HEParameterSet<CKKS>(N=32768, Q=60+40+41+...+41 bits, P=60+60+60+60+60+60 bits,
    //                      total=825/881 bits, digits=2, scale=2^40, security=sec128)
    auto context = parameters.make_context();
    context->generate();
    double scale = params.scale();

For CKKS, ``Q`` holds a first prime of ``first_modulus_bit_size`` bits (60 by default) and one prime per level close to the scale. Rescaling by the prime :math:`q_l` of level :math:`l` turns the scale :math:`s_l` into :math:`s_l^2 / q_l` when ciphertexts of the same scale are multiplied, so :math:`q_l` is the prime closest to :math:`s_l^2 / 2^{\Delta}`: the scale stays close to :math:`2^{\Delta}` at every level. ``P`` gets as many 60-bit primes as the limit leaves room for, up to the number of ``Q`` primes: a larger ``P`` splits ``Q`` into fewer key switching digits, which makes key switching faster and the keys smaller. Of these it takes the fewest that give the same number of digits. ``special_modulus_count`` fixes the number of ``P`` primes instead.

For BFV, ``plain_modulus_bit_size`` (20 by default) chooses the largest prime of that many bits that is 1 mod :math:`2N`, which batching needs; ``plain_modulus`` gives :math:`t` directly. ``Q`` gets the bits the noise of ``multiplicative_depth`` multiplications needs plus a margin of 10 bits, from a model calibrated against ``remainder_noise_budget``: a fresh ciphertext uses about :math:`\log_2 t + \log_2 N / 2 + 2` bits and each multiplication about :math:`\log_2 t + \log_2 N` more. A multiplication by a plaintext can use as much as one by a ciphertext. The ``P`` primes have a bit more than the ``Q`` primes, and as for CKKS there are as many as the limit leaves room for, up to the number of ``Q`` primes, and of these the fewest that give the same number of digits; ``special_modulus_count`` fixes their number instead. For example, depth 4 with a 20-bit :math:`t` gets three ``Q`` and three ``P`` primes at :math:`N = 2^{14}`, one digit instead of three, and depth 30 ten ``P`` primes for 19 ``Q`` primes at :math:`N = 2^{16}`, two digits instead of 19, which makes a relinearization or Galois key about 6.5 times smaller.

``poly_modulus_degree`` fixes :math:`N`, and ``security_level`` selects 128-, 192- or 256-bit security. ``sec_level_type::none`` drops the security limit, for experiments: it needs ``poly_modulus_degree``, and ``P`` then gets one prime per ``Q`` prime, a single digit, unless ``special_modulus_count`` gives another number. When the request does not fit, ``generate()`` throws ``ParameterError`` with the largest depth that fits. ``HEParameterSet::apply(context)`` sets the parameters of a context created by hand.

``HEParams`` also takes the moduli themselves, as bit sizes or as primes, which gives the same context as setting them by hand:

.. code-block:: cpp

    heongpu::HEParams<heongpu::Scheme::CKKS> params;
    params.coeff_modulus_bit_sizes = {60, 40, 40};
    params.special_modulus_bit_sizes = {60};
    auto context = params.make_context();  // the smallest N whose limit fits them

``coeff_modulus`` and ``special_modulus`` give the primes, which need ``poly_modulus_degree``. With explicit moduli, ``multiplicative_depth``, ``first_modulus_bit_size`` and ``special_modulus_count`` must keep their defaults. The parameters can still be set directly on the context with ``set_poly_modulus_degree`` and ``set_coeff_modulus_bit_sizes`` or ``set_coeff_modulus_values``.

The following table, derived from the lattice-estimator, shows the recommended total bit-length of the key modulus (:math:`\tilde{Q} = QP`) for different polynomial degrees to achieve a 128-bit security level.

.. list-table:: Recommended Modulus Sizes for 128-bit Security
   :widths: 25 25
   :header-rows: 1

   * - Polynomial Degree (:math:`N`)
     - Total Modulus Bit-Length (:math:`\log_2 \tilde{Q}`)
   * - :math:`2^{12}` (4096)
     - 109
   * - :math:`2^{13}` (8192)
     - 218
   * - :math:`2^{14}` (16384)
     - 438
   * - :math:`2^{15}` (32768)
     - 881
   * - :math:`2^{16}` (65536)
     - 1761

Levels and Depth (CKKS)
-----------------------

A CKKS ciphertext is counted in two directions. Its **depth** is the number of primes that rescaling or ``mod_drop`` has dropped from :math:`Q`: 0 at the top level. Its **level** is the number of rescalings left: :math:`L` at the top, for the :math:`L` scaling primes after the first prime of :math:`Q`, and 0 when only the first prime is left. So ``level() == L - depth()``. The encoder and ``Plaintext`` take a depth (``encode(message, scale, depth)``), while the context and bootstrapping take levels (``get_level_scale(level)``, ``BootstrappingLevels``). ``Ciphertext`` has both ``depth()`` and ``level()``.

Automatic Rescaling (CKKS)
--------------------------

By default, the application rescales CKKS ciphertexts after multiplications and brings operands to the same level with ``mod_drop``. In the automatic rescale mode the operators do both:

.. code-block:: cpp

    heongpu::HEParams<heongpu::Scheme::CKKS> params;
    params.multiplicative_depth = 3;
    params.rescale_mode = heongpu::rescale_mode::automatic;
    auto context = params.make_context();
    context->generate();
    // ... keys, encoder, encryptor, decryptor and operators

    auto plaintext = encoder.encode(values);  // the scale of the top level
    auto x = encryptor.encrypt(plaintext);
    auto x2 = operators.multiply(x, x);
    operators.relinearize_inplace(x2, relin_key);
    auto x3 = operators.multiply(x2, x);  // x2 is rescaled, x brought down to its level
    operators.relinearize_inplace(x3, relin_key);
    auto sum = operators.add(x3, x);      // x brought down to the level of x3

A context created by hand selects the mode after setting its moduli with ``context->set_rescale_mode(heongpu::rescale_mode::automatic, scale)``.

**Scales of the levels.** In the automatic mode every level :math:`l` has one scale :math:`s_l`: the top level has the scale given to the mode, and :math:`s_{l-1} = s_l^2 / q_l`, where :math:`q_l` is the prime that rescaling drops at level :math:`l`. Multiplying two ciphertexts of level :math:`l` and rescaling gives exactly :math:`s_{l-1}`, so all ciphertexts of a level have the same scale and can be added. ``context->get_level_scale(level)`` returns :math:`s_l`. The primes that ``HEParams`` chooses keep every :math:`s_l` next to the requested scale.

**What the operators do.**

* A ciphertext that needs a rescaling is rescaled before a multiplication, a rotation, a conjugation or an addition to a ciphertext that does not need one; two ciphertexts that both need one are added as they are, which saves rescalings when products are summed. ``rescale_inplace`` does nothing on a ciphertext that does not need a rescaling, so code written for the manual mode gives the same results.
* A ciphertext at a higher level is brought down to the level of the other operand with the scale of that level: it is dropped to the level above, multiplied by an integer and rescaled. ``mod_drop`` does the same.
* ``HEEncoder::encode(message)`` encodes at the top level and ``encode(message, depth)`` at a depth, both with the scale of the level. Plaintexts must have the scale of their level; a plaintext brought down to a lower level keeps its scale, so encode it at the depth of the ciphertext.
* ``multiply_plain(ciphertext, constant, ...)`` encodes the constant with the scale of the level; its ``scale`` argument is not used.
* ``evaluate_polynomial`` and ``evaluate_chebyshev`` return the result with the scale of its level.
* The inputs are not changed; the operators rescale and bring down copies.

Relinearization is not automatic: it needs the relinearization key. Two operands at the same level with different scales, for example a ciphertext encoded with another scale, raise ``ParameterError``. The CoeffToSlot and SlotToCoeff transforms and the logic operations of ``HELogicOperator`` without bootstrapping need the manual mode and raise ``StateError`` in the automatic one.

**Bootstrapping.** The levels above the bootstrapping results are those the bootstrapping uses, so in the automatic mode the scale belongs to the level of the results: ``context->set_rescale_mode(heongpu::rescale_mode::automatic, preset.scale, preset.levels().output_level)``. The levels above this top level have no scale, ``encode(message)`` encodes at the top level, and the encryptor encrypts a plaintext at its level. The four bootstrappings then take ciphertexts that need a rescaling, and their results have the level and scale of the top level; ``generate_bootstrapping_params`` raises ``ParameterError`` if the top level or its scale differ.

The mode is not part of the context identity and is not saved with the context, since keys and ciphertexts are the same in both modes; set it again after ``load``.

Sessions (HESession)
--------------------

``HESession`` holds one party's whole computation: the context of the parameters, the secret, public and relinearization keys, the encoder, the encryptor, the decryptor and the operators. Its ciphertexts, ``SessionCiphertext``, have the operators ``+``, ``-`` and ``*``:

.. code-block:: cpp

    heongpu::HEParams<heongpu::Scheme::CKKS> params;
    params.multiplicative_depth = 3;
    heongpu::HESession<heongpu::Scheme::CKKS> session(params);

    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    auto z = x * x * y + 0.5 * x - y;
    std::vector<double> result = session.decrypt(z);

* A product of two ciphertexts is relinearized with the key of the session.
* For CKKS the session uses the automatic rescale mode, unless ``params.rescale_mode`` is set to ``rescale_mode::manual``; a session made from an ``HEParameterSet`` takes its mode. In the manual mode, products are not rescaled: rescale them with ``session.operators().rescale_inplace(z.ciphertext())``.
* The other operand can be a constant or a vector of values, one per slot. CKKS takes ``double`` values and encodes them at the level of the ciphertext with the scale the operation needs. BFV takes ``int64_t`` constants and ``uint64_t`` or ``int64_t`` vectors, modulo the plain modulus.
* ``x.rotate(k)`` rotates the slots (both rows for BFV), ``x.conjugate()`` conjugates them (CKKS) and ``x.rotate_columns()`` swaps the two rows (BFV). The Galois key of a step is generated the first time it is used; ``HESession(params, {1, 2, 4})`` generates the keys of the given steps up front, and ``session.add_rotations(steps)`` adds keys later.
* ``x.rotate_hoisted({1, 2, 3})`` returns the rotations by several steps, which share the decomposition of ``x``; ``x.sum_slots()`` returns the sum of the slots in every slot, and ``x.inner_product(y)`` the inner product with a ciphertext or a vector.
* ``heongpu::NOT(a)``, ``AND(a, b)``, ``OR``, ``XOR``, ``NAND``, ``NOR`` and ``XNOR`` compute logic gates of bits (0 or 1) in the slots with the arithmetic of the session, so they also work in the automatic rescale mode; a gate of two inputs uses one multiplication.
* CKKS: ``session.evaluate_polynomial`` and ``session.evaluate_chebyshev``, see above; ``session.decrypt_complex`` returns complex slots. BFV: ``session.decrypt_signed`` returns the slots from :math:`-(t-1)/2` to :math:`(t-1)/2`, and ``session.noise_budget(x)`` the bits of noise budget left.
* ``session.context()``, ``session.secret_key()``, ``session.relin_key()``, ``session.galois_key()``, ``session.operators()`` and the others return the objects of the lower level API, and ``x.ciphertext()`` the ``Ciphertext``. ``session.wrap(ciphertext)`` turns a ``Ciphertext`` encrypted with the keys of the session into a ``SessionCiphertext``.

**Bootstrapping.** A CKKS session made from a bootstrapping preset generates the bootstrapping keys up front and bootstraps its ciphertexts:

.. code-block:: cpp

    heongpu::HESession<heongpu::Scheme::CKKS> session(
        heongpu::CKKSBootstrappingPreset::regular(10));  // 10 levels between bootstrappings
    auto x = session.encrypt(xs);
    for (int i = 0; i < 30; i++)
    {
        if (x.levels_left() < 2)
            x = x.bootstrap();
        x = 0.5 * x * x + 0.25;
    }

* The session uses the automatic rescale mode with the level of the bootstrapping results as its top level; ``encrypt`` encrypts at that level.
* ``x.levels_left()`` returns the levels the computation can still use before ``x`` needs a bootstrapping. Bootstrapping is never started by itself.
* ``x.bootstrap()`` uses the regular, slim or bit bootstrapping of the preset; a gate preset has ``session.AND_bootstrapping(x, y)``, ``OR_bootstrapping``, ``XOR_bootstrapping``, ``NAND_bootstrapping``, ``NOR_bootstrapping`` and ``XNOR_bootstrapping`` instead.
* The relinearization and Galois keys of the session are those of the bootstrapping, so rotations by their steps need no new keys. The keys and the bootstrapping matrices of ``regular()`` take about 11 GB of GPU memory.

``session.on_stream(stream)`` returns a view of the session that runs on a CUDA stream: it shares the keys, its ciphertexts belong to the stream, and an operation runs on the stream of its left operand. A thread can thus work on its own stream:

.. code-block:: cpp

    auto worker = session.on_stream(stream);   // waits for the work queued on the session's stream
    auto y = worker.encrypt(values) * shared;  // on stream
    std::vector<double> result = worker.decrypt(y);

As with the lower level API, a ciphertext of another stream must be ready before it is used; ``session.synchronize()`` waits for the operations queued on the stream of a session.

Copies of a session share its state, and threads can use them at the same time. A ``SessionCiphertext`` keeps the state of its session, so it stays usable after the session object is destroyed. Operations on ciphertexts of two different sessions raise ``ContextMismatchError``.

**Roles, saving and loading.** A session made from parameters or a preset is the *owner* of its keys. ``save`` writes it with the keys of a role, and ``load`` reads it:

.. code-block:: cpp

    // The owner sends the keys of the evaluator, without the secret key, and a ciphertext.
    owner.save(keys_file, heongpu::session_role::evaluator);
    owner.encrypt(xs).save(input_file);

    // The server computes and cannot decrypt.
    auto server = heongpu::HESession<heongpu::Scheme::CKKS>::load(
        keys_file, heongpu::session_role::evaluator);
    auto x = server.load_ciphertext(input_file);
    (x * x).rotate(1).save(output_file);

    std::vector<double> result = owner.decrypt(owner.load_ciphertext(output_file));

* ``session_role::owner`` saves all keys, ``evaluator`` the public, relinearization, Galois and bootstrapping keys, and ``encryptor`` the public key only. ``session.role()`` returns the role of a session.
* A session without the secret key raises ``StateError`` for ``decrypt``, and ``KeyNotFoundError`` for a rotation by a step it has no key for, since it cannot generate one: the owner adds the step with ``add_rotations`` and saves the session again. An encryptor raises ``KeyNotFoundError`` for products of ciphertexts.
* ``load(stream, role)`` raises ``StateError`` before reading the keys unless the session was saved with that role, so a server never holds a secret key it was sent by mistake.
* The rescale mode, the scale and the bootstrapping parameters are saved with the session. The keys of the evaluator of ``CKKSBootstrappingPreset::regular()`` take several GB.

Storage Management with `ExecutionOptions`
------------------------------------------

The ``ExecutionOptions`` struct provides precise control over where data resides, which is critical for optimizing complex computational pipelines by minimizing host-device data transfers.

* ``set_storage_type(storage_type)``: Defines the desired location (``device`` or ``host``) for the *output* data of an operation, and for the object an ``*_inplace`` operation changes.
* ``set_initial_location(bool)``: Only applies to the object an ``*_inplace`` operation changes.

The inputs of an operation are taken by ``const`` reference and never change: an input in host memory is read from a temporary copy on the device and stays in host memory. To keep an input on the device for later operations, move it there once with ``store_in_device()``.

.. code-block:: cpp

    // Inputs in host memory, the output on the device.
    ExecutionOptions options;
    options.set_storage_type(storage_type::device);

    operators.add(host_input1, host_input2, device_output, options);
    // host_input1 and host_input2 are still in host memory.

Serialization
-------------

HEonGPU includes a high-performance serialization module for all major objects (Context, keys, plaintexts, ciphertexts). This enables fast disk I/O and client-server transfers. The module supports:

* **Raw Binary Format**: For maximum speed.
* **Zlib Compression**: An optional compressed format that can reduce storage and bandwidth requirements by up to 60%, useful for network-bound applications or long-term storage.

Since HEonGPU 1.3, ``save`` writes the binary format version 2 by default. It starts with a 16 byte header: the magic number ``HEGP``, the format version, the object type, the scheme and the identity of the context the object belongs to (see `Context Identity`_). ``load`` checks the header and throws ``SerializationError`` for data that is not an HEonGPU object of the expected type and scheme, and ``ContextMismatchError`` for data of another context.

``load`` also reads the format version 1 written by HEonGPU 1.2 and earlier. HEonGPU 2.0 writes the format version 2 only, which HEonGPU 1.3 and later read.

Error Handling
--------------

HEonGPU reports errors with exceptions. The most common ones have their own types, declared in ``<heongpu/util/error.h>``, and their messages contain the values involved and how to fix the problem.

.. list-table:: HEonGPU exception types
   :widths: 25 25 50
   :header-rows: 1

   * - Type
     - Standard base
     - Thrown when
   * - ``ParameterError``
     - ``std::invalid_argument``
     - A parameter or an argument has an invalid value, for example a scale out of bounds.
   * - ``StateError``
     - ``std::invalid_argument``
     - An object is used before it is ready, for example a key that is not generated yet.
   * - ``ContextMismatchError``
     - ``std::invalid_argument``
     - An object is used with a different ``HEContext`` (see below).
   * - ``LevelMismatchError``
     - ``std::invalid_argument``
     - The operands are not at the levels the operation requires.
   * - ``KeyNotFoundError``
     - ``std::invalid_argument``
     - A key needed by the operation is missing, for example a Galois key for a rotation step.
   * - ``SerializationError``
     - ``std::runtime_error``
     - Serialized data is malformed or written for another scheme.
   * - ``SystemError``
     - ``std::runtime_error``
     - A system service fails, for example the random number generator.

All of them also derive from ``heongpu::Error``, so one handler can catch every HEonGPU exception. Existing handlers for the standard types keep working.

.. code-block:: cpp

    try
    {
        operators.add(ciphertext1, ciphertext2, result);
    }
    catch (const heongpu::LevelMismatchError& error)
    {
        // "Ciphertexts leveled are not equal (depth 0 and 1)! ..."
        operators.mod_drop_inplace(ciphertext1);
        operators.add(ciphertext1, ciphertext2, result);
    }
    catch (const heongpu::Error& error)
    {
        std::cerr << error.what() << std::endl;
    }

Context Identity
----------------

Every generated ``HEContext`` has an identity, returned by ``context_id()``, computed from the scheme, the ring size, the Q and P primes and (for BFV) the plaintext modulus. Ciphertexts, plaintexts and keys remember the identity of the context they were created with, and operators, encryptors, decryptors and encoders throw ``ContextMismatchError`` when they get an object of another context. This catches, for example, two contexts with the same ring size and number of primes but different primes, which would otherwise give wrong results.

Contexts generated from the same parameters, for example on a client and a server, have the same identity, so their objects can be used together. The serialization format version 2 stores the identity, so loaded objects are checked as well: when they are loaded into an object of another context, and when they are used later. Data in the format version 1 has no identity: an object loaded from it keeps the identity of the context it was created with, and a default constructed object loaded from it is not checked this way.

Tutorials
---------

Integer Arithmetic with the BFV Scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BFV scheme is ideal for applications requiring exact computations on encrypted integers. A typical multiplication involves:

1.  **Encrypt**: Encrypt two plaintexts containing integers.
2.  **Multiply**: Use the ``HEArithmeticOperator`` to multiply the two ciphertexts. Homomorphic multiplication increases the size of the resulting ciphertext from 2 to 3 polynomials.
3.  **Relinearize**: Relinearization is a key-switching operation that reduces the ciphertext back to its original size of 2 polynomials, making subsequent operations more efficient. This step requires ``RelinearizationKeys``. The library will not permit further multiplications on a non-relinearized ciphertext.

Approximate Arithmetic with the CKKS Scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CKKS scheme is tailored for applications involving real or complex numbers, such as privacy-preserving machine learning. Key steps include:

1.  **Encode**: Use the ``HEEncoder`` to encode a vector of doubles into a ``Plaintext``. This requires specifying an initial `scale` factor (e.g., :math:`2^{40}`), which determines the precision of the approximation.
2.  **Encrypt**: Encrypt the plaintext.
3.  **Multiply**: Perform a homomorphic multiplication. The scale of the resulting ciphertext will be approximately the product of the input scales (e.g., :math:`(2^{40})^2 = 2^{80}`).
4.  **Rescale**: To manage the precision and prevent the scale from growing uncontrollably, a `rescale` operation is performed. This operation divides the internal plaintext by one of the coefficient moduli, effectively reducing the scale (e.g., back down to ~:math:`2^{40}`) and consuming one level of the modulus chain. The library will not permit further multiplications until a rescale operation is performed.

Operations on a ciphertext and a plaintext need both at the same depth. To use a plaintext with a ciphertext that has lost moduli, encode it at the depth of the ciphertext directly, instead of encoding it and calling ``mod_drop``:

.. code-block:: cpp

    encoder.encode(plaintext, message, scale, ciphertext.depth());
    operators.add_plain_inplace(ciphertext, plaintext);

Noise Management in BFV
^^^^^^^^^^^^^^^^^^^^^^^

In the BFV scheme, HEonGPU provides a mechanism for **invariant noise estimation**. This allows users to monitor the health of a ciphertext. The library can compute the remaining "noise budget," expressed in bits, which quantifies the remaining capacity for noise growth while still ensuring correct decryption.

Correct decryption is guaranteed as long as the noise polynomial :math:`v` in the expression :math:`\frac{t}{Q}(ct \cdot sk) = m + v + at` has coefficients with an absolute value less than :math:`1/2`. The noise budget starts at approximately :math:`\log_2(Q/t)` and decreases with each operation. Once the budget reaches 0, the ciphertext becomes too noisy to be decrypted correctly. Users can query this budget to determine how many more operations can be safely performed.

.. note::
    For the CKKS scheme, HEonGPU does not provide direct noise estimation. Users must manually monitor the depth of the circuit by counting the number of coefficient moduli that have been consumed by rescale operations.
