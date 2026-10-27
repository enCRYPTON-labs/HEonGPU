.. _advanced_topics:

Advanced Topics
===============

This section covers advanced usage patterns, configuration options, and integration strategies for developers looking to leverage the full capabilities of the HEonGPU library.

Configuration (``defines.h``)
-----------------------------

The ``src/include/heongpu/kernel/defines.h`` file is an essential configuration header that contains key settings defining the library's limits and default behaviors. Advanced users can modify these values to tune the library for specific hardware or demanding use cases.

Key settings include:

* **Polynomial Degree**: ``MAX_POLY_DEGREE`` (default: 65536) and ``MIN_POLY_DEGREE`` (default: 4096) define the supported range for polynomial degrees.
* **Modulus Bit-Length**: These constants specify the valid bit-lengths for user-defined and general modulus values, ensuring cryptographic security.
* **Galois Key Capability**: ``MAX_SHIFT`` (default: 8) sets the rotation range of the default Galois keys, ``Galoiskey(context)``: the steps 1, 2, 4, ..., 2^(MAX_SHIFT - 1) and their negatives. To use another range without rebuilding the library, construct the key with ``Galoiskey(context, max_shift)``, which generates the steps up to 2^max_shift.
* **Memory Pool Sizes**: The initial and maximum sizes for the device (GPU) and host (CPU) memory pools are defined as percentages of available system memory. By default, the GPU pool is initialized to 50% of VRAM and can grow to 80%, while the pinned host memory pool is initialized to 10% of RAM and can grow to 20%. These values can be adjusted for systems with different memory capacities or for applications with particularly large memory footprints.

Process-Wide State
------------------

HEonGPU keeps some state for the whole process. This matters when it is embedded into a larger application:

* **Memory pool**: One pool per process, created by the first ``generate()``. The ``MemoryPoolConfig`` of that call sets its sizes; ``use_memory_pool`` of the most recent call decides whether new allocations use it.
* **Random number generator**: One generator per process, seeded by the first ``generate()``.
* **CUDA kernel stack size**: HEonGPU's kernels need at least 2048 bytes per thread. ``generate()`` raises a lower ``cudaLimitStackSize`` and keeps a higher one. With ``MemoryPoolConfig::raise_kernel_stack_size = false``, it does not change the limit and throws ``StateError`` if the limit is too low.
* **RMM default resource**: Not changed. HEonGPU allocates from its own resource. With ``MemoryPoolConfig::set_as_rmm_default = true``, HEonGPU's pool also becomes the process-wide RMM default; the previous default is restored when a later context turns it off, and at exit.

HEonGPU does not change other CUDA settings and does not select the CUDA device; it uses the current device of the calling thread.

HEonGPU writes no output on its own. Only ``print_parameters()`` and ``MemoryPool::print_memory_pool_status()`` print, when called, to the stream they are given (``std::cout`` by default); errors are reported with exceptions. RMM, which HEonGPU uses for device memory, can log to ``stderr``, for example when an allocation fails. Its log level is set with ``rmm::default_logger().set_level(rapids_logger::level_enum::off)`` (from ``<rmm/logger.hpp>``) or another level.

With the static library, every shared library or plugin that embeds HEonGPU has its own copy of this state. With the shared library (``HEonGPU_BUILD_SHARED=ON``), the whole process shares one.

The first ``generate()`` sets this state up. An application can also control it explicitly:

.. code-block:: cpp

    heongpu::initialize(config);   // optional; generate() does it otherwise
    // ... use HEonGPU ...
    heongpu::shutdown();           // releases the memory pool and the generator
    heongpu::initialize(other);    // sets it up again, with another configuration

``shutdown()`` requires all HEonGPU objects (contexts, keys, plaintexts, ciphertexts, encoders, encryptors, decryptors and operators) to be destroyed; otherwise it throws ``StateError`` and releases nothing. It also restores the RMM default resource if ``set_as_rmm_default`` replaced it.

Versioning and Compatibility
----------------------------

HEonGPU follows `semantic versioning <https://semver.org/>`_:

* **Major versions** (2.0, 3.0, ...) can change or remove APIs.
* **Minor versions** (1.4, 1.5, ...) add features without breaking code that compiles against the previous minor version. They can change the ABI, so rebuild code that uses HEonGPU; the shared library is versioned accordingly (``libheongpu.so.1.4``).
* **Patch versions** (1.4.1, ...) fix bugs without changing the API or the ABI.

APIs that will be removed are first marked ``[[deprecated]]``, with the replacement in the warning, for at least one minor version, and removed in the next major version.

Deprecated since 1.7.0, removed in 2.0:

* CKKS ``rotate_rows``, ``rotate_rows_inplace`` and ``rotate_rows_hoisted``: use ``rotate``, ``rotate_inplace`` and ``rotate_hoisted``. CKKS slots are not split into rows; BFV keeps ``rotate_rows`` and ``rotate_columns``.
* The CKKS bootstrapping functions that take the switch keys by pointer (``regular_bootstrapping``, ``slim_bootstrapping``, ``bit_bootstrapping`` and the gates such as ``AND_bootstrapping``): use the overloads with ``CKKSBootstrappingKeys``.
* ``serialization_format::v1`` for writing; 2.0 still reads data in this format.
* ``keyswitching_type::KEYSWITCHING_METHOD_I``: every context uses hybrid key switching. 2.0 removes ``keyswitching_type`` from the public API.
* The upper case values ``storage_type::HOST``/``DEVICE``, ``encoding::SLOT``/``COEFFICIENT``, ``arithmetic_bootstrapping_type::NONE``/``REGULAR_BOOTSTRAPPING``/``SLIM_BOOTSTRAPPING`` and ``logic_bootstrapping_type::NONE``/``BIT_BOOTSTRAPPING``/``GATE_BOOTSTRAPPING``: use the lower case values (``storage_type::device``, ``encoding::slot``, ``arithmetic_bootstrapping_type::regular``, ...).
* ``bootstrapping_key_indexs``: use ``bootstrapping_key_indices`` (since 1.2).

2.0 will also take the inputs of the operations by ``const`` reference, which does not change code that calls them.

Serialized data stays readable: HEonGPU 1.x reads the data written by every earlier 1.x version. A newer version can write data that older versions cannot read; ``save(stream, heongpu::serialization_format::v1)`` writes data for HEonGPU 1.2 and earlier until 2.0; it is deprecated since 1.7.0, and 2.0 keeps reading such data.

Supported toolchain: CUDA 12.8 or newer, a C++20 compiler (tested with GCC 11.4) and CMake 4.0 or newer.

**Thread safety**

* A generated context is not changed afterwards and can be shared by threads.
* Keys and operators can be shared by threads that run operations on their own ciphertexts and plaintexts and their own CUDA streams, as the multi-stream examples do.
* An object that an operation writes, or moves between host and device memory, must not be used by another thread at the same time.
* The memory pool is thread-safe. Call ``heongpu::initialize()`` and ``heongpu::shutdown()`` while no other thread uses HEonGPU.
* Key generation and encryption draw from one process-wide random number generator that is not synchronized: run them from one thread at a time.

Multiparty Computation (MPC)
----------------------------

HEonGPU includes support for secure **Multi-Party Computation (MPC)** protocols, enabling multiple parties to collaboratively compute on encrypted data. This is achieved through Multiparty Homomorphic Encryption (MHE) capabilities, providing a framework for threshold encryption models such as `N-out-of-N`.

A key feature of the MPC implementation is **Collective (Distributed) Bootstrapping** for both the BFV and CKKS schemes. This powerful technique, based on designs by Mouchet et al. and Balle et al., allows a group of parties to jointly refresh a ciphertext, resetting its noise level without a trusted dealer. The implementation merges the share creation and re-encryption steps into a single, efficient CUDA kernel launch. This allows deep multi-party workloads to continue running entirely on the GPU without pausing to reset noise, which is critical for complex, collaborative privacy-preserving applications.

Using HEonGPU in a Downstream CMake Project
-------------------------------------------

To integrate HEonGPU into your own CMake project, first ensure the library is installed on your system. The installed library provides the necessary config files to make integration seamless. In your project's ``CMakeLists.txt``, you can use ``find_package`` to locate and link against the library.

.. code-block:: cmake

    # Set the project language to include CUDA
    project(<your-project> LANGUAGES CXX CUDA)

    # Find the CUDA Toolkit, which is a dependency
    find_package(CUDAToolkit REQUIRED)
    
    # ... your other project configurations ...

    # Find the HEonGPU package
    find_package(HEonGPU REQUIRED)

    # ... define your executable target ...
    add_executable(<your-target> main.cpp)

    # Link your application against the HEonGPU library and the CUDA runtime
    target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu CUDA::cudart)

    # Enable separable compilation for CUDA, which is often required
    set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

This applies to the default static library (``libheongpu.a``), which leaves the CUDA device link to your project. HEonGPU can also be built as a shared library with ``-D HEonGPU_BUILD_SHARED=ON`` (``libheongpu.so``). The shared library does its own device link, so your project does not need to enable CUDA or separable compilation, and all libraries in a process that use it share one HEonGPU, including its memory pool:

.. code-block:: cmake

    project(<your-project> LANGUAGES CXX)
    find_package(HEonGPU REQUIRED)
    target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu)

The shared library is versioned with the major and minor version (for example ``libheongpu.so.1.4``), because its ABI can change between minor versions. Its kernels are internal and cannot be launched from outside the library.

Project Roadmap
---------------

HEonGPU is an actively developing project with a clear vision for the future. The roadmap indicates a strategic expansion from accelerating core cryptographic primitives to enabling complex, end-to-end privacy-preserving systems.

Planned enhancements include:

* **Support for Additional Schemes**: The library plans to add support for the **BGV** scheme to expand its cryptographic capabilities.
* **Python Wrapper**: To make the library more accessible to the data science and machine learning communities, a Python wrapper is on the roadmap.
* **Multi-GPU Support**: The architecture is planned to be extended to support multi-GPU configurations, including a multi-GPU memory pool structure, to facilitate the execution of even larger-scale applications.
