.. _getting_started:

Getting Started
===============

This section provides a practical guide to installing the HEonGPU library, verifying the installation by running tests and examples, and executing your first encrypted computation. The process is designed to be straightforward for developers familiar with C++ and CMake in a Linux environment.

Prerequisites
-------------

Before building the library, ensure your development environment meets the following requirements. The library handles most of its third-party C++ dependencies (like `GPU-NTT`, `RMM`, `Thrust`, and `GoogleTest`) automatically via CMake.

**System Dependencies:**

* **CMake**: Version 4.0 or higher
* **GCC**: A modern C++ compiler with C++20 support (tested with GCC 11.4)
* **GMP**: The GNU Multiple Precision Arithmetic Library
* **NTL**: A Library for doing Number Theory
* **CUDA Toolkit**: Version 12.2 or higher
* **OpenSSL**: Version 1.1.1 or higher
* **ZLIB**: The data compression library

On Ubuntu, the system packages are installed with ``sudo apt install libgmp-dev libntl-dev libssl-dev zlib1g-dev``.

Building the Library
--------------------

The library uses a standard CMake build process. The most critical configuration step is specifying the correct compute capability for your target GPU, as this is essential for both correctness and performance.

**Step 1: Clone the Repository**
    First, obtain the source code from the official GitHub repository.

    .. code-block:: bash

        git clone https://github.com/Alisah-Ozcan/HEonGPU.git
        cd HEonGPU

**Step 2: Configure with CMake**
    Run CMake from the root of the repository to generate the build files. You **must** set the ``CMAKE_CUDA_ARCHITECTURES`` variable to match your GPU's architecture. This flag instructs the CUDA compiler (``nvcc``) to generate code specifically optimized for your hardware.

    .. code-block:: bash

        cmake -S . -D CMAKE_CUDA_ARCHITECTURES=XX -B build

    Replace ``XX`` with the appropriate value from the table below. For example, for an NVIDIA RTX 4090 (Ada Lovelace architecture), you would use ``89``.

    .. list-table:: GPU Architecture to CMAKE_CUDA_ARCHITECTURES Mapping
       :widths: 25 25
       :header-rows: 1

       * - GPU Architecture
         - Compute Capability
       * - Volta
         - 70, 72
       * - Turing
         - 75
       * - Ampere
         - 80, 86
       * - Ada Lovelace
         - 89
       * - Hopper
         - 90
       * - Blackwell
         - 100, 120

**Step 3: Compile the Library**
    Once configuration is complete, build the library using the following command. The ``-jN`` flag will use multiple cores to speed up compilation. Replace ``N`` by the number of your logical cores (i.e. the output of ``nproc``).

    .. code-block:: bash

        cmake --build ./build/ -jN

**Step 4: Install (Optional)**
    To install the library system-wide (e.g., in `/usr/local/lib` and `/usr/local/include`), run the install command.

    .. code-block:: bash

        sudo cmake --install build

    To install somewhere else, for example without ``sudo``, pass a prefix, and point your project to it with ``-D CMAKE_PREFIX_PATH=<dir>``. The installed package can be moved to another directory afterwards.

    .. code-block:: bash

        cmake --install build --prefix $HOME/.local

Verifying the Installation
--------------------------

After a successful build, you can verify the library's functionality by running the built-in tests, benchmarks, and examples. To do this, you must enable them during the CMake configuration step.

* **To build and run tests**:
    .. code-block:: bash

        # Configure with tests enabled
        cmake -S . -D HEonGPU_BUILD_TESTS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
        
        # Build
        cmake --build ./build/ -j
        
        # Run all tests
        cmake --build build --target test

* **To build and run examples**:
    .. code-block:: bash

        # Configure with examples enabled
        cmake -S . -D HEonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
        
        # Build
        cmake --build ./build/ -j
        
        # Run a specific example
        ./build/bin/examples/1_basic_bfv

Your First Encrypted Computation
---------------------------------

The following "toy example" from the repository demonstrates a complete FHE workflow: setting up the context, generating keys, encoding, encrypting, performing a homomorphic addition, decrypting, and decoding the result.

.. code-block:: cpp
   :linenos:

    #include <heongpu/heongpu.hpp>

    int main() {     
        // 1. Set up the HEContext for the BFV scheme
        heongpu::HEContext<heongpu::Scheme::BFV> context =
                heongpu::GenHEContext<heongpu::Scheme::BFV>();

        // 2. Define and set encryption parameters
        size_t poly_modulus_degree = 8192;
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_default_values(1); // Use 1 default prime for the coeff modulus
        int plain_modulus = 1032193;
        context->set_plain_modulus(plain_modulus);
        context->generate(); // Finalize context and pre-compute values on GPU

        // 3. Generate keys
        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        // 4. Create Encoder, Encryptor, Decryptor, and Operator objects
        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context, public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context, secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context, encoder);

        // 5. Create a message and encode it into a plaintext
        std::vector<uint64_t> message(poly_modulus_degree, 8ULL);
        heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
        encoder.encode(P1, message);

        // 6. Encrypt the plaintext into a ciphertext
        heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
        encryptor.encrypt(C1, P1);

        // 7. Perform a homomorphic operation (in-place addition)
        operators.add_inplace(C1, C1); // Result: 8 + 8 = 16

        // 8. Decrypt the result
        heongpu::Plaintext<heongpu::Scheme::BFV> P2(context);
        decryptor.decrypt(P2, C1);

        // 9. Decode the plaintext to retrieve the final vector
        std::vector<uint64_t> result;
        encoder.decode(result, P2);

        // The 'result' vector should now contain the value 16 in all its slots.
        return 0;
    }
