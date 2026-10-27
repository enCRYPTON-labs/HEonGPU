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
* **CUDA Toolkit**: Version 12.8 or higher
* **OpenSSL**: Version 1.1.1 or higher
* **ZLIB**: The data compression library

On Ubuntu, the system packages are installed with ``sudo apt install libgmp-dev libntl-dev libssl-dev zlib1g-dev``.

Building the Library
--------------------

The library uses a standard CMake build process. The most critical configuration step is specifying the correct compute capability for your target GPU, as this is essential for both correctness and performance.

**Step 1: Clone the Repository**
    First, obtain the source code from the official GitHub repository.

    .. code-block:: bash

        git clone --recursive https://github.com/Alisah-Ozcan/HEonGPU.git
        cd HEonGPU

    GPU-NTT, GPU-FFT and RNGonGPU are git submodules. If the repository was cloned without ``--recursive``, fetch them with ``git submodule update --init --recursive``.

**Step 2: Configure with CMake**
    Run CMake from the root of the repository to generate the build files. You **must** set the ``CMAKE_CUDA_ARCHITECTURES`` variable to match your GPU's architecture. This flag instructs the CUDA compiler (``nvcc``) to generate code specifically optimized for your hardware.

    .. code-block:: bash

        cmake -S . -D CMAKE_CUDA_ARCHITECTURES=XX -B build

    Replace ``XX`` with the appropriate value from the table below. For example, for an NVIDIA RTX 4090 (Ada Lovelace architecture), you would use ``89``.

    Configuring downloads RMM, and GoogleTest when the tests are enabled and no installed GoogleTest is found. To build without network access, install RMM 26.08 (and GoogleTest for the tests), pass their location with ``-D CMAKE_PREFIX_PATH=<dir>`` and set ``-D HEonGPU_USE_SYSTEM_RMM=ON``.

    When HEonGPU builds RMM from source, installing HEonGPU also installs this RMM, because the installed HEonGPU needs it. If another RMM version is installed at the same prefix, configuring warns that it would be replaced. ``-D HEonGPU_USE_SYSTEM_RMM=ON`` uses an installed RMM 26.08 and never downloads RMM; ``-D HEonGPU_INSTALL_RMM=OFF`` does not install the RMM built from source, so projects that use the installed HEonGPU have to provide RMM 26.08 themselves.

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
        // 1. Choose the encryption parameters for the BFV scheme: one
        //    multiplication in a row with a 20 bit plaintext modulus at
        //    128-bit security
        heongpu::HEParams<heongpu::Scheme::BFV> params;
        params.multiplicative_depth = 1;
        params.plain_modulus_bit_size = 20;

        // 2. Set up the HEContext with them
        heongpu::HEContext<heongpu::Scheme::BFV> context = params.make_context();
        context->generate(); // Finalize context and pre-compute values on GPU
        size_t poly_modulus_degree = context->get_poly_modulus_degree();

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
