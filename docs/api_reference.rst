.. _api_reference:

API Reference
=============

This section provides a high-level overview of the public C++ API of HEonGPU. The design is intended to be stable and intuitive, abstracting all underlying CUDA complexity and drawing inspiration from the user-friendly design of Microsoft SEAL. The API is organized within the ``heongpu`` namespace and is heavily templated to support different FHE schemes.

Core Enumerations
-----------------

**Scheme Selection**
    Used to specify the desired homomorphic encryption scheme during context creation.

    .. code-block:: cpp

        namespace heongpu {
            enum class Scheme { 
                BFV, 
                CKKS, 
                TFHE 
            };
        }

**Key-Switching Method**
    Key-switching operations like relinearization and rotation use hybrid
    key switching (`KEYSWITCHING_METHOD_II`). The P modulus base count sets
    the number of primes per digit; with `P_size == 1` every digit is one
    prime, which is the classical method (`KEYSWITCHING_METHOD_I` of earlier
    versions, whose data still loads). `P_size == 0` is invalid.

    .. code-block:: cpp

        namespace heongpu {
            enum class keyswitching_type {
                KEYSWITCHING_METHOD_I,  // Earlier versions, for one P prime; deprecated
                KEYSWITCHING_METHOD_II, // Hybrid method, used by every context
            };
        }

Context and Parameter Classes
-----------------------------

**HEContext**
    The central class that manages the entire cryptographic environment. It must be the first object instantiated. It holds the encryption parameters, validates them, and pre-computes necessary data on the GPU.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEContext {
        public:
            // Constructor optionally takes security level.
            HEContext(sec_level_type sec_level = sec_level_type::sec128);

            // Setters for core encryption parameters.
            void set_poly_modulus_degree(size_t degree);
            void set_coeff_modulus_default_values(int level);
            void set_plain_modulus(int plain_modulus); // For BFV

            // Finalizes the context and uploads pre-computations to the GPU.
            void generate();
        };

**rescale_mode** (CKKS)
    ``rescale_mode::manual`` (the default) leaves rescaling and bringing operands to the same level to the application; with ``rescale_mode::automatic`` the operators do both, see Automatic Rescaling in the User Guide.

    .. code-block:: cpp
       :linenos:

        // On HEContextImpl<Scheme::CKKS>, after the moduli are set:
        void set_rescale_mode(rescale_mode mode, double scale = 0.0);
        rescale_mode get_rescale_mode() const;
        double get_level_scale(int level) const;  // automatic mode

**HEParams**
    Chooses the ring size and the primes from what the computation needs, see the Parameter Selection Guide. ``generate()`` returns them as an ``HEParameterSet``, which ``apply(context)`` sets on a context and ``make_context()`` sets on a new one.

    .. code-block:: cpp
       :linenos:

        template <> struct HEParams<Scheme::CKKS> {
            int multiplicative_depth = 1;
            int scale_bit_size = 40;
            int first_modulus_bit_size = 60;
            sec_level_type security_level = sec_level_type::sec128;
            std::size_t poly_modulus_degree = 0;  // 0: the smallest that fits
            int special_modulus_count = 0;        // 0: as many as fit

            HEParameterSet<Scheme::CKKS> generate() const;
            HEContext<Scheme::CKKS> make_context() const;
            double scale() const;
            // Unset: manual for generate(), automatic for HESession.
            std::optional<rescale_mode> rescale_mode;
        };

        template <> struct HEParams<Scheme::BFV> {
            int multiplicative_depth = 1;
            Data64 plain_modulus = 0;             // 0: chosen from the bit size
            int plain_modulus_bit_size = 20;
            sec_level_type security_level = sec_level_type::sec128;
            std::size_t poly_modulus_degree = 0;  // 0: the smallest that fits

            HEParameterSet<Scheme::BFV> generate() const;
            HEContext<Scheme::BFV> make_context() const;
        };

Key Management Classes
----------------------

**HEKeyGenerator**
    Generates all necessary cryptographic keys from a valid ``HEContext``.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEKeyGenerator {
        public:
            HEKeyGenerator(const HEContext<scheme_type>& context);

            void generate_secret_key(Secretkey<scheme_type>& sk);
            void generate_public_key(Publickey<scheme_type>& pk, const Secretkey<scheme_type>& sk);
            void generate_relin_key(Relinkey<scheme_type>& rlk, const Secretkey<scheme_type>& sk);
            void generate_galois_key(/* ... */); // For rotations
        };

**Key Storage Classes**
    These classes are containers for the key data, which resides primarily on the GPU.

    .. code-block:: cpp

        template <Scheme scheme_type> class Secretkey;
        template <Scheme scheme_type> class Publickey;
        template <Scheme scheme_type> class Relinkey;
        template <Scheme scheme_type> class Galoiskey;

Data and Cryptographic Classes
------------------------------

**Plaintext and Ciphertext**
    These classes hold the plaintext (encoded) and ciphertext data.

    .. code-block:: cpp

        template <Scheme scheme_type> class Plaintext;
        template <Scheme scheme_type> class Ciphertext;

**HEEncoder**
    Handles the conversion of user data into the plaintext polynomial format required for encryption.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEEncoder {
        public:
            HEEncoder(const HEContext<scheme_type>& context);

            // BFV: Encodes a vector of integers.
            void encode(Plaintext<Scheme::BFV>& ptxt, const std::vector<uint64_t>& message);

            // CKKS: Encodes a vector of doubles with a given scale.
            void encode(Plaintext<Scheme::CKKS>& ptxt, const Message<Scheme::CKKS>& msg, double scale);
            
            // Corresponding decode functions are also provided.
            void decode(std::vector<uint64_t>& message, const Plaintext<Scheme::BFV>& ptxt);
            void decode(Message<Scheme::CKKS>& msg, const Plaintext<Scheme::CKKS>& ptxt);
        };

**HEEncryptor and HEDecryptor**
    Perform the core cryptographic transformations.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEEncryptor {
        public:
            HEEncryptor(const HEContext<scheme_type>& context, const Publickey<scheme_type>& pk);
            void encrypt(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);
        };

        template <Scheme scheme_type>
        class HEDecryptor {
        public:
            HEDecryptor(const HEContext<scheme_type>& context, const Secretkey<scheme_type>& sk);
            void decrypt(Plaintext<scheme_type>& ptxt, const Ciphertext<scheme_type>& ctxt);
        };

**HEArithmeticOperator**
    The primary class for executing all homomorphic operations on the GPU.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEArithmeticOperator {
        public:
            HEArithmeticOperator(const HEContext<scheme_type>& context, const HEEncoder<scheme_type>& encoder);

            // In-place and out-of-place arithmetic operations
            void add(const Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2, Ciphertext<scheme_type>& result);
            void add_inplace(Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2);
            void multiply(const Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2, Ciphertext<scheme_type>& result);
            void multiply_inplace(Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2);
            
            // Operations with plaintexts
            void add_plain_inplace(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);
            void multiply_plain_inplace(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);

            // Key-switching and modulus management operations
            void relinearize_inplace(Ciphertext<scheme_type>& ctxt, const Relinkey<scheme_type>& rlk);
            void rotate_inplace(Ciphertext<Scheme::CKKS>& ctxt, const Galoiskey<Scheme::CKKS>& gk, int steps);  // CKKS
            void rotate_rows_inplace(Ciphertext<Scheme::BFV>& ctxt, const Galoiskey<Scheme::BFV>& gk, int steps);  // BFV, both rows
            void rescale_inplace(Ciphertext<Scheme::CKKS>& ctxt); // CKKS only
        };

**HESession and SessionCiphertext**
    One party's computation in one object, see Sessions in the User Guide. ``SessionCiphertext`` has the operators ``+``, ``-``, ``*``, their compound assignments and unary ``-``, with other ciphertexts of the session, constants and vectors.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HESession {
        public:
            explicit HESession(const HEParams<scheme_type>& params, const std::vector<int>& rotations = {});
            explicit HESession(const HEParameterSet<scheme_type>& parameters, const std::vector<int>& rotations = {});

            SessionCiphertext<scheme_type> encrypt(const std::vector<double>& values) const;  // CKKS; BFV: uint64_t, int64_t
            std::vector<double> decrypt(const SessionCiphertext<scheme_type>& ctxt) const;   // CKKS; BFV: uint64_t
            SessionCiphertext<scheme_type> wrap(Ciphertext<scheme_type> ctxt) const;
            void add_rotations(const std::vector<int>& steps) const;

            // CKKS: decrypt_complex, evaluate_polynomial, evaluate_chebyshev, scale, get_rescale_mode
            // BFV: decrypt_signed, noise_budget, plain_modulus

            // The objects of the lower level API.
            HEContext<scheme_type> context() const;
            Relinkey<scheme_type>& relin_key() const;
            Galoiskey<scheme_type>& galois_key() const;
            HEArithmeticOperator<scheme_type>& operators() const;
            // ... secret_key, public_key, encoder, encryptor, decryptor
        };

        template <Scheme scheme_type>
        class SessionCiphertext {
        public:
            Ciphertext<scheme_type>& ciphertext();
            SessionCiphertext rotate(int steps) const;
            SessionCiphertext conjugate() const;       // CKKS
            SessionCiphertext rotate_columns() const;  // BFV
        };
