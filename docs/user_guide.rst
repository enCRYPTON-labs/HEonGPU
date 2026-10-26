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

Parameter Selection Guide
-------------------------

Choosing the right encryption parameters is crucial for balancing security, performance, and the computational depth of the circuits you can evaluate. In HEonGPU, all parameters are explicitly set at the ``HEContext`` level before key generation.

* **Security Level**: The library is designed and tested for a 128-bit security level, which is the standard for most modern applications.
* **Polynomial Modulus Degree (N)**: This parameter is the primary driver of security and performance. Larger values of :math:`N` provide more security and a larger "noise budget" (allowing for more computations), but at a significant performance cost. The library is optimized for power-of-two values for :math:`N` in the range of :math:`[2^{12}, 2^{16}]`.
* **Coefficient Modulus (q)**: This is a chain of prime numbers whose product forms the ciphertext modulus. The total bit-size of :math:`q` determines the computational depth—a larger modulus allows for more sequential multiplications. The library's use of the Residue Number System (RNS) makes operations with these prime chains highly suitable for parallelization on the GPU.
* **Plaintext Modulus (t)**: For the BFV scheme, this parameter defines the size of the integer message space (i.e., computations are performed modulo :math:`t`).

The library provides default parameter generation to simplify this process. The following table, derived from the lattice-estimator, shows the recommended total bit-length of the key modulus (:math:`\tilde{Q} = QP`) for different polynomial degrees to achieve a 128-bit security level.

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

Storage Management with `ExecutionOptions`
------------------------------------------

The ``ExecutionOptions`` struct provides precise control over where data resides, which is critical for optimizing complex computational pipelines by minimizing host-device data transfers.

* ``set_storage_type(storage_type)``: Defines the desired location (``DEVICE`` or ``HOST``) for the *output* data of an operation.
* ``set_initial_location(bool)``: If ``true``, ensures that input data returns to its original location after the computation. If ``false``, input data that was moved to the device for computation will remain on the device.

.. code-block:: cpp

    // Example: Inputs are on HOST, but we want the output on DEVICE
    // and want inputs to remain on HOST after the operation.
    ExecutionOptions options;
    options.set_storage_type(storage_type::DEVICE)
           .set_initial_location(true);

    operators.add(host_input1, host_input2, device_output, options);

The following table details the behavior for all combinations:

.. list-table:: ExecutionOptions Behavior
   :widths: 15 15 25 25 20
   :header-rows: 1

   * - `set_storage_type`
     - `set_initial_location`
     - Input Locations (Before)
     - Input Locations (After)
     - Output Location
   * - `DEVICE`
     - `true`
     - `input1: HOST, input2: HOST`
     - `input1: HOST, input2: HOST`
     - `DEVICE`
   * - `DEVICE`
     - `false`
     - `input1: HOST, input2: HOST`
     - `input1: DEVICE, input2: DEVICE`
     - `DEVICE`
   * - `HOST`
     - `true`
     - `input1: HOST, input2: DEVICE`
     - `input1: HOST, input2: HOST`
     - `HOST`
   * - `HOST`
     - `false`
     - `input1: DEVICE, input2: DEVICE`
     - `input1: DEVICE, input2: DEVICE`
     - `HOST`

Serialization
-------------

HEonGPU includes a high-performance serialization module for all major objects (Context, keys, plaintexts, ciphertexts). This enables fast disk I/O and client-server transfers. The module supports:

* **Raw Binary Format**: For maximum speed.
* **Zlib Compression**: An optional compressed format that can reduce storage and bandwidth requirements by up to 60%, useful for network-bound applications or long-term storage.

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

Contexts generated from the same parameters, for example on a client and a server, have the same identity, so their objects can be used together. The serialization format does not store the identity yet, so objects loaded with ``load`` are not checked this way.

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

Noise Management in BFV
^^^^^^^^^^^^^^^^^^^^^^^

In the BFV scheme, HEonGPU provides a mechanism for **invariant noise estimation**. This allows users to monitor the health of a ciphertext. The library can compute the remaining "noise budget," expressed in bits, which quantifies the remaining capacity for noise growth while still ensuring correct decryption.

Correct decryption is guaranteed as long as the noise polynomial :math:`v` in the expression :math:`\frac{t}{Q}(ct \cdot sk) = m + v + at` has coefficients with an absolute value less than :math:`1/2`. The noise budget starts at approximately :math:`\log_2(Q/t)` and decreases with each operation. Once the budget reaches 0, the ciphertext becomes too noisy to be decrypted correctly. Users can query this budget to determine how many more operations can be safely performed.

.. note::
    For the CKKS scheme, HEonGPU does not provide direct noise estimation. Users must manually monitor the depth of the circuit by counting the number of coefficient moduli that have been consumed by rescale operations.
