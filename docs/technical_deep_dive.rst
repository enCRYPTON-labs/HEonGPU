.. _technical_deep_dive:

Technical Deep Dive
===================

This section explores the advanced technical details of HEonGPU, focusing on the architectural decisions, algorithmic choices, and low-level optimization strategies that enable its high performance. The content is primarily derived from the extensive research detailed in the project's associated academic publications.

Library Architecture
--------------------

The performance of HEonGPU is rooted in its **"all-in-on-the-GPU"** design philosophy. Unlike hybrid models that may offload only the most intensive kernels, HEonGPU is designed to keep the entire computational workflow resident on the GPU. This strategy is a direct response to a critical observation: for many FHE workloads, the system is not purely compute-bound but becomes **memory-bound** due to the large size of ciphertexts. The latency of transferring these multi-megabyte data structures between host (CPU) and device (GPU) memory can easily negate any speedup gained from GPU computation. By keeping data resident on the GPU, HEonGPU minimizes this costly overhead.

This architecture is complemented by the use of **multiple CUDA streams**, which allows the GPU's scheduler to overlap memory transfers with computation and execute independent kernels concurrently. This maximizes hardware utilization and overall throughput, as detailed in the main architectural paper `ePrint 2024/1543 <https://eprint.iacr.org/2024/1543>`_.

Accelerating Polynomial Arithmetic: The Number Theoretic Transform (NTT)
------------------------------------------------------------------------

The most computationally intensive operation in lattice-based FHE is the multiplication of large-degree polynomials. HEonGPU accelerates this using the **Number Theoretic Transform (NTT)**, which allows two polynomials to be multiplied in quasi-linear time, :math:`O(N \log N)`. The research behind HEonGPU involved a systematic investigation into the most efficient GPU implementations of the NTT, with a focus on optimizing memory access patterns. Two primary algorithms were implemented and optimized:

* **radix2-CT (Merge) NTT**: Based on the recursive Cooley-Tukey FFT algorithm, this method processes the input vector in :math:`\log_2 N` sequential outer loop iterations.
* **4-Step NTT**: This algorithm reframes the 1D NTT as a series of smaller, independent 2D NTTs on a matrix representation of the polynomial coefficients.

The core of the optimization lies in a novel strategy for selecting CUDA launch parameters (kernel count, block size, and block shape) to create the most efficient memory access patterns for a given polynomial degree.

GPU-Specific NTT Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NTT implementation in HEonGPU is not a direct port of CPU algorithms but a ground-up redesign for GPU architecture, as detailed in `ePrint 2023/1410 <https://eprint.iacr.org/2023/1410>`_. Key optimization strategies include:

* **Optimizing Kernel Launches**: Each kernel launch incurs significant overhead. The library minimizes launches by partitioning the :math:`\log_2 N` stages of the NTT across an optimal number of kernels, rather than launching a new kernel for each stage. This involves a trade-off: fewer kernels reduce launch overhead, but sometimes an additional kernel can enable a more favorable memory access pattern, leading to a net performance gain.
* **Coalesced Memory Access**: Kernels are structured so that threads within a warp (a group of 32 threads) access consecutive memory locations. This is achieved by carefully shaping thread blocks and grids into two dimensions, ensuring that threads processing related data are physically grouped to maximize memory bandwidth.
* **Efficient Shared Memory Use**: Fast, on-chip shared memory is used as a programmable cache. For stages of the NTT where threads within a block need to exchange data, that data is first loaded from slow global memory into fast shared memory, operated upon, and then written back. This dramatically reduces costly global memory traffic.
* **Optimal Block Size and Occupancy**: While a CUDA thread block can contain up to 1024 threads, using the maximum is not always optimal. A smaller block size (e.g., 256 threads) can lead to higher *occupancy*—the ratio of active warps to the maximum possible on a Streaming Multiprocessor (SM). Higher occupancy allows the GPU to hide memory latency more effectively by switching to other active warps while one is stalled waiting for data. Experiments showed that a block size of 256 often provides the best balance of parallelism and resource utilization.

Key-Switching Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Key-switching is a fundamental primitive used in operations like relinearization (after multiplication) and rotations. It is computationally expensive and has been a major focus of optimization. Two key-switching methods are analyzed in `ePrint 2025/124 <https://eprint.iacr.org/2025/124>`_:

* **Method I (Fan-Vercauteren)**: The classical approach: every prime of the ciphertext modulus is one digit of the decomposition.
* **Method II (Hybrid Key Switching)**: Groups the primes into digits of several primes, which reduces the number of NTTs and Hadamard products compared to Method I, at the cost of extra base conversion operations. This method is generally faster on GPUs where the NTT acceleration is significant.

Method I is Method II with digits of one prime, so HEonGPU implements one hybrid key switching. The digits have as many primes as the P modulus; with one P prime this is Method I, with the kernels specialized for it.

Memory Management with RAPIDS Memory Manager (RMM)
----------------------------------------------------

To handle the frequent allocation and deallocation of large memory blocks for ciphertexts and keys, HEonGPU integrates the **RAPIDS Memory Manager (RMM)**.

* **Memory Pooling**: Instead of making expensive `cudaMalloc` and `cudaFree` calls for every object, RMM is used to create a large memory pool on the GPU at initialization (`rmm::pool_memory_resource`). Subsequent allocations are then serviced rapidly from this pool, reducing overhead and memory fragmentation.
* **Pinned Host Memory**: For the CPU-side memory pool, `rmm::mr::pinned_host_memory_resource` is used. Pinned memory is non-pageable and allows for much faster, asynchronous data transfers between the host and device.
* **Custom Vector Types**: The library uses custom vector classes, `heongpu::DeviceVector` and `heongpu::HostVector`, which are integrated with the RMM memory pools to provide a standard C++ vector-like interface while benefiting from the optimized memory management backend.

Decomposition of Homomorphic Operations
----------------------------------------

A high-level homomorphic operation, such as a BFV ciphertext multiplication, is decomposed into a sequence of distinct, individually optimized CUDA kernels. A typical workflow for multiplication includes:

1.  **Base Conversion Kernels**: As part of the Residue Number System (RNS) machinery, these kernels convert polynomial coefficients between different modular bases.
2.  **NTT Kernel**: Transforms the input ciphertexts into the evaluation domain using the highly optimized NTT implementation.
3.  **Element-wise Multiplication Kernel**: A simple but massively parallel kernel that performs the multiplication of the transformed polynomials.
4.  **Inverse NTT (INTT) Kernel**: Transforms the result back from the evaluation domain to the coefficient domain.
5.  **Relinearization Kernels**: The relinearization process itself is a complex operation involving key-switching, which is further decomposed into its own sequence of base conversions, NTTs, and other arithmetic kernels.

This modular decomposition allows the library to achieve massive parallelism and apply specific optimizations at every stage of the computation.
