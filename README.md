# 🚀 **HEonGPU** - A GPU Based Homomorphic Encryption Library

HEonGPU is a high-performance library designed to optimize Fully Homomorphic Encryption (FHE) operations on GPUs. By leveraging the parallel processing power of GPUs, it significantly reduces the computational load of FHE through concurrent execution of complex operations. Its multi-stream architecture enables efficient parallel processing and minimizes the overhead of data transfers between the CPU and GPU. These features make HEonGPU ideal for large-scale encrypted computations, offering reduced latency and improved performance.

The goal of HEonGPU is to provide:
- A high-performance framework for executing FHE schemes, specifically `BFV`, `CKKS` and `TFHE`, by leveraging the parallel processing capabilities of CUDA.
- A user-friendly C++ interface that requires no prior knowledge of GPU programming, with all CUDA kernels encapsulated in easy-to-use classes.
- An optimized multi-stream architecture that ensures efficient memory management and concurrent execution of encrypted computations on the GPU.

For more information about HEonGPU: 
- https://eprint.iacr.org/2024/1543
- https://heongpu.readthedocs.io/

### Current HEonGPU Capabilities and Schemes

<div align="center">

| Capability / Scheme            | HEonGPU   |
|:------------------------------:|:---------:|
| BFV                            | ✓         |
| CKKS                           | ✓         |
| BGV                            | Very Soon |
| TFHE                           | ✓         |
| CKKS Regular Bootstrapping     | ✓         |
| CKKS Slim Bootstrapping        | ✓         |
| CKKS Bit Bootstrapping         | ✓         |
| CKKS Gate Bootstrapping        | ✓         |
| TFHE Gate Bootstrapping        | ✓         |
| Multiparty Computation (MPC)   | ✓         |
| Collective Bootstrapping (MPC) | ✓         |

</div>

## News

### 🚨 **New Feature: [Collective(Distributed) Bootstrapping](example/mpc/)**

The HEonGPU library now delivers `Collective Bootstrapping` for both BFV and CKKS, drawing on the designs introduced by [Mouchet et al.](https://eprint.iacr.org/2020/304.pdf)  and [Balle et al.](https://arxiv.org/pdf/2009.00349) A streamlined CUDA path merges share creation and re-encryption into a single launch, allowing deep multi-party workloads to keep running entirely on the GPU without pausing to reset noise.


### 🚨 **New Scheme: [TFHE (Torus Fully Homomorphic Encryption)](example/basic/15_basic_tfhe.cpp)**

The HEonGPU library now includes support for the `TFHE` (Torus Fully Homomorphic Encryption) scheme with GPU acceleration. This enables efficient evaluation of Boolean circuits using fast gate bootstrapping and low-latency parallel execution on modern CUDA-enabled GPUs.

Currently, the implementation supports a fixed parameter set targeting 128-bit security. In upcoming releases, we plan to:

- Make the parameters fully configurable,
- Provide default parameter sets for **128-bit**, **192-bit**, and **256-bit** security levels.
- Further optimize and accelerate the TFHE implementation with improved CUDA kernels and parallelism strategies,
- Introduce native support for **homomorphic unsigned arithmetic** via new types:  
  `huint8`, `huint16`, `huint32`, `huint64`, `huint128`, and `huint256`.


<div align="center">

|           | uint8 | uint16 | uint32 | uint64 | uint128 | uint256 |
|-----------|-------|--------|--------|--------|---------|---------|
| [TFHE-rs](https://github.com/zama-ai/tfhe-rs)   | 31.53 | 31.54  | 31.55  | 32.03  | 33.74   | 58.32   |
| [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931)  | 18.63 | 18.61  | 18.87  | 24.23  | 29.97   | 58.30   |
| **HEonGPU**   | **12.72** | **12.75**  | **13.60**  | **15.88**  | **23.10**   | **38.24**   |
| Speedup (vs [TFHE-rs](https://github.com/zama-ai/tfhe-rs))  | 2.48× | 2.47×  | 2.32×  | 2.02×  | 1.46×   | 1.52×   |
| Speedup (vs [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931)) | 1.46× | 1.46×  | 1.39×  | 1.53×  | 1.30×   | 1.52×   |

</div>

> **Table**: Latency (ms) comparison between [TFHE-rs](https://github.com/zama-ai/tfhe-rs), [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931), and HEonGPU for different bit sizes (on GPU), based on the STD128 parameter set.  
> All benchmarks were performed on an **NVIDIA RTX 4090 GPU**.  
> The last two rows show the speedup of HEonGPU with respect to [TFHE-rs](https://github.com/zama-ai/tfhe-rs) and [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931).

### 🚨 **New Feature: Serialization Module**

The new serializer module has been successfully integrated into HEonGPU. It provides high-performance serialization and deserialization of homomorphic encryption objects (Context, Secretkey, Publickey, Relinkey, Galoiskey, Plaintext, Ciphertext, etc.) in raw binary or optional [Zlib-compressed](https://zlib.net/) formats. This enhancement enables blazing-fast disk I/O, seamless client-server transfers, and up to a 60% reduction in storage and bandwidth.

### 🚨 **New Feature: Integration [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU)**

[RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU) has been successfully integrated into HEonGPU. [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU) features a secure Deterministic Random Bit Generator (DRBG) designed according to NIST [Recommendation for Random Number Generation Using Deterministic Random Bit Generators](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf). This integration enhances GPU-based random number generation, ensuring both high performance and robust security.

### 🚨 **New Application: Private Information Retrieval on GPU ([PIRonGPU](https://github.com/Alisah-Ozcan/PIRonGPU))**

[PIRonGPU](https://github.com/Alisah-Ozcan/PIRonGPU) is a high-performance library that enhances secure data retrieval through Private Information Retrieval (PIR) on GPUs. By modifying the  [SealPIR](https://github.com/microsoft/SealPIR) protocol with **HEonGPU**, it achieves rapid, confidential querying, offering an efficient and scalable solution for privacy-sensitive applications.

### 🚨 **New Feature: Logic Operation and [3 More CKKS Bootstrapping Types](example/bootstrapping/README.md)**

HEonGPU now provides comprehensive support for logic operations across both the `BFV` and `CKKS` encryption schemes. In addition, the latest update introduces three new `CKKS` Bootstrapping types; two of which leverage `Bit` Bootstrapping and `Gate` Bootstrapping techniques, while the third employs `Slim` Bootstrapping, a method that is significantly more efficient than `Regular` Bootstrapping.  These enhancements not only broaden HEonGPU’s functionality but also significantly improve its performance in managing noise and enabling efficient, secure computations on GPU platforms.

The Logic Operations supported: 
- NOT, AND, NAND, OR, NOR, XOR, XNOR

3 More CKKS Bootstrapping Types:
- `Slim` Bootstrapping (supports only Real Numbers)
- `Bit` Bootstrapping (supports only Binary Numbers)
- `Gate` Bootstrapping (supports only Binary Numbers)
  - AND, NAND, OR, NOR, XOR, XNOR

<div align="center">

### Execution times of the HEonGPU Bootstrapping Operations (on RTX 4090)

<table>
  <tr>
    <th>Bootstrapping Type</th>
    <th>N</th>
    <th>Slot Count</th>
    <th>Remaining Level</th>
    <th>Total Time</th>
    <th>Amortized Time</th>
    <th>Precision (bits)</th>
  </tr>
  <tr>
    <td rowspan="1"><b>Regular Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>109.28 ms</td>
    <td>3.33 µs</td>
    <td>17.95</td>
  </tr>
  <tr>
    <td rowspan="3"><b>Slim Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>38.55 ms</td>
    <td>1.18 µs</td>
    <td>20.47</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>70.59 ms</td>
    <td>2.15 µs</td>
    <td>20.47</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>15 Level</td>
    <td>94.58 ms</td>
    <td>2.89 µs</td>
    <td>20.47</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Bit Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>50.72 ms</td>
    <td>1.55 µs</td>
    <td>27.40</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>8 Level</td>
    <td>82.43 ms</td>
    <td>2.52 µs</td>
    <td>27.40</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Gate Bootstrapping*</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>62.90 ms</td>
    <td>1.92 µs</td>
    <td>27.66</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>7 Level</td>
    <td>99.45 ms</td>
    <td>3.04 µs</td>
    <td>27.67</td>
  </tr>
</table>
</div>


<div align="center">

### Execution times of the HEonGPU Bootstrapping Operations (on RTX 5090)

<table>
  <tr>
    <th>Bootstrapping Type</th>
    <th>N</th>
    <th>Slot Count</th>
    <th>Remaining Level</th>
    <th>Total Time</th>
    <th>Amortized Time</th>
    <th>Precision (bits)</th>
  </tr>
  <tr>
    <td rowspan="1"><b>Regular Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>63.51 ms</td>
    <td>1.94 µs</td>
    <td>17.94</td>
  </tr>
  <tr>
    <td rowspan="3"><b>Slim Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>23.92 ms</td>
    <td>0.73 µs</td>
    <td>20.47</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>40.03 ms</td>
    <td>1.22 µs</td>
    <td>20.48</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>15 Level</td>
    <td>56.45 ms</td>
    <td>1.72 µs</td>
    <td>20.58</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Bit Bootstrapping</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>35.70 ms</td>
    <td>1.09 µs</td>
    <td>27.40</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>8 Level</td>
    <td>52.13 ms</td>
    <td>1.59 µs</td>
    <td>27.40</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Gate Bootstrapping*</b></td>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>40.35 ms</td>
    <td>1.23 µs</td>
    <td>27.67</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>7 Level</td>
    <td>59.50 ms</td>
    <td>1.82 µs</td>
    <td>27.67</td>
  </tr>
</table>
</div>

Dense secret key (h=192) with a temporary sparse secret key (h=32), median of 5 runs. Remaining Level is the number of levels left before the next bootstrapping.

*: Timing is for all gates; precision is measured using the AND example.

### 🚨 **New Feature: [CKKS Regular Bootstrapping](example/bootstrapping/README.md)**

HEonGPU now includes support for `CKKS Regular` Bootstrapping, enabling efficient evaluation of deep computational circuits with high precision and security. On an NVIDIA RTX 4090, it performs `CKKS` Regular Bootstrapping for 
N=65536 in about 110 ms.

### 🚨 **New Feature: [Multiparty Computation (MPC) Support](example/mpc/README.md)**

HEonGPU now includes support for **Multiparty Computation (MPC)** protocols, providing a secure and collaborative framework for encrypted computations. By incorporating *Multiparty Homomorphic Encryption (MHE)* capabilities, the library enables distributed computations with threshold encryption models such as `N-out-of-N`. The implementation is fully optimized for GPU environments, delivering minimal latency and maximum performance in collaborative settings.

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=4.0
- [GCC](https://gcc.gnu.org/) with C++20 support (tested with GCC 11.4)
- [GMP](https://gmplib.org/)
- [NTL](https://libntl.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=12.8
- [OpenSSL](https://www.openssl.org/) >= 1.1.1
- [ZLIB](https://zlib.net/)

On Ubuntu: `sudo apt install libgmp-dev libntl-dev libssl-dev zlib1g-dev`.

### Third-Party Dependencies
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
- [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT)
- [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU)
- [RMM](https://github.com/rapidsai/rmm)
- [GoogleTest](https://github.com/google/googletest)

HEonGPU builds GPU-NTT, GPU-FFT and RNGonGPU from its git submodules. It downloads RMM unless `HEonGPU_USE_SYSTEM_RMM` is set, and GoogleTest unless an installed one is found.
Without `CMAKE_CUDA_ARCHITECTURES` (or the `CUDAARCHS` environment variable), HEonGPU builds for the architecture of the first GPU of the machine. Set `-D CMAKE_CUDA_ARCHITECTURES=<...>` to build for other or several GPUs.

### Build & Install

To build and install HEonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

<div align="center">

| GPU Architecture | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value) |
|:----------------:|:---------------------------------------------------:|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada Lovelace | 89 |
| Hopper | 90 |
| Blackwell | 100, 120 |

</div>

```bash
$ git clone --recursive https://github.com/Alisah-Ozcan/HEonGPU.git
$ cd HEonGPU
$ cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
$ cmake --build ./build/
$ sudo cmake --install build
```
GPU-NTT, GPU-FFT and RNGonGPU are git submodules. If the repository was cloned without `--recursive`, fetch them with `git submodule update --init --recursive`.

Configuring downloads RMM, and GoogleTest when the tests are enabled and no installed GoogleTest is found. To build without network access, install RMM 26.08 (and GoogleTest for the tests), pass their location with `-D CMAKE_PREFIX_PATH=<dir>` and set `-D HEonGPU_USE_SYSTEM_RMM=ON`.

When HEonGPU builds RMM from source, installing HEonGPU also installs this RMM (with CCCL, rapids_logger and nvtx3), because the installed HEonGPU needs it. If another RMM version is installed at the same prefix, configuring warns that it would be replaced. Two options control this:
- `-D HEonGPU_USE_SYSTEM_RMM=ON`: use an installed RMM 26.08 and fail if there is none, never download RMM.
- `-D HEonGPU_INSTALL_RMM=OFF`: do not install the RMM built from source. Projects that use the installed HEonGPU then need to provide RMM 26.08 themselves.

Available build types: `Debug`, `Release` (default), `RelWithDebInfo`, `MinSizeRel`. Override with `-D CMAKE_BUILD_TYPE=<type>`. These propagate to all bundled libraries (GPU-FFT, GPU-NTT, RNGonGPU).

The default install prefix is `/usr/local`. To install somewhere else, for example without `sudo`, set it when configuring (`-D CMAKE_INSTALL_PREFIX=<dir>`) or when installing (`cmake --install build --prefix <dir>`), and pass the same directory to your project with `-D CMAKE_PREFIX_PATH=<dir>`. The installed package can be moved to another directory afterwards.

### Uninstall

To uninstall HEonGPU and thirdparty libraries:

```bash
$ cd build
$ sudo make uninstall
```

The uninstall script will remove:
- **HEonGPU** library and headers
- **GPU-NTT**, **GPU-FFT**, **RNGonGPU** libraries and headers
- All CMake configuration files

## Testing & Benchmarking

To run tests:

```bash
$ cmake -S . -D HEonGPU_BUILD_TESTS=ON -B build -D CMAKE_BUILD_TYPE=Debug
$ cmake --build ./build/

$ ./build/bin/test/<...>
$ Example: ./build/bin/test/bfv_addition_testcases
```
Or:
```
$ cmake -S . -D HEonGPU_BUILD_TESTS=ON -B build -D CMAKE_BUILD_TYPE=Debug
$ cmake --build ./build/
$ cmake --build build --target test
```

To run benchmarks:
```bash
$ cmake -S . -D HEonGPU_BUILD_BENCHMARKS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build -D CMAKE_BUILD_TYPE=Release
$ cmake --build ./build/

$ ./build/bin/benchmark/<...>
$ Example: ./build/bin/benchmark/bfv_benchmark
```

## Examples

To run examples:

```bash
$ cmake -S . -D HEonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build -D CMAKE_BUILD_TYPE=Release
$ cmake --build ./build/

$ ./build/bin/examples/<...>
$ Example: ./build/bin/examples/1_basic_bfv
```

### Toy Example

`HESession` holds one party's whole computation: the context of the parameters, the keys and the operators.

```c++
#include <heongpu/heongpu.hpp>

int main() {
    // Parameters for one multiplication in a row with a 20 bit plaintext
    // modulus at 128-bit security.
    heongpu::HEParams<heongpu::Scheme::BFV> params;
    params.multiplicative_depth = 1;
    heongpu::HESession<heongpu::Scheme::BFV> session(params);

    std::vector<uint64_t> message(session.slot_count(), 8ULL);
    auto C1 = session.encrypt(message);
    auto C2 = C1 + C1;
    std::vector<uint64_t> result = session.decrypt(C2);

    return 0;
}
```

A CKKS session made from a bootstrapping preset bootstraps its ciphertexts, and a session saved with the keys of the evaluator computes on a server without the secret key:

```c++
heongpu::HESession<heongpu::Scheme::CKKS> session(
    heongpu::CKKSBootstrappingPreset::regular(10));  // 10 levels between bootstrappings
auto x = session.encrypt(values);
for (int i = 0; i < 30; i++) {
    if (x.levels_left() < 2)
        x = x.bootstrap();
    x = 0.5 * x * x + 0.25;
}

session.save(keys, heongpu::session_role::evaluator);  // without the secret key
auto server = heongpu::HESession<heongpu::Scheme::CKKS>::load(
    keys, heongpu::session_role::evaluator);
```

The same BFV computation with the lower level API, whose objects give full control over the keys, the CUDA streams and the memory of every operation:

```c++
#include <heongpu/heongpu.hpp>

int main() {   
    // Parameters for one multiplication in a row with a 20 bit plaintext
    // modulus at 128-bit security.
    heongpu::HEParams<heongpu::Scheme::BFV> params;
    params.multiplicative_depth = 1;
    params.plain_modulus_bit_size = 20;
    heongpu::HEContext<heongpu::Scheme::BFV> context = params.make_context();
    context->generate();
    size_t poly_modulus_degree = context->get_poly_modulus_degree();

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context, encoder);

    std::vector<uint64_t> message(poly_modulus_degree, 8ULL);
    heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
    encoder.encode(P1, message);

    heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
    encryptor.encrypt(C1, P1);

    operators.add_inplace(C1, C1);

    heongpu::Plaintext<heongpu::Scheme::BFV> P2(context);
    decryptor.decrypt(P2, C1);

    std::vector<uint64_t> result;
    encoder.decode(result, P2);

    return 0;
}
```


## Configuration Header ([define.h](src/include/heongpu/kernel/defines.h))

The [define.h](src/include/heongpu/kernel/defines.h) file is an essential configuration file for HEonGPU, containing key settings that define the library's limits and capabilities, including polynomial degrees, modulus bit-lengths, and memory pool sizes. 

Features in [define.h](src/include/heongpu/kernel/defines.h):
- **Polynomial Degree:** `MAX_POLY_DEGREE` (65536) and `MIN_POLY_DEGREE` (4096) define the range for polynomial degrees used in FHE.

- **Modulus Bit-Length:** `MAX_USER_DEFINED_MOD_BIT_COUNT` (60), `MIN_USER_DEFINED_MOD_BIT_COUNT` (30), `MAX_MOD_BIT_COUNT` (61), `MIN_MOD_BIT_COUNT` (30) specify valid bit-lengths for user-defined and general modulus values.

- **Maximum BFV Auxiliary Base Count:** `MAX_BSK_SIZE` (64) sets the maximum size for the BFV auxiliary base.

- **Galois Key Capability:** `MAX_SHIFT` (8) sets the rotation range of the default Galois keys, `Galoiskey(context)`. To use another range without rebuilding the library, construct the key with `Galoiskey(context, max_shift)`.

- **Device**:
  - Initial size (`0.5`): 50% of available GPU memory.
  - Max size (`0.95`): 95% of available GPU memory.
- **Host**:
  - Initial size (default): 100 MB when no runtime config is provided.
  - Max size (`0.4`): 40% of available CPU memory.

The device pool grows on demand from the initial size up to the max size, so the
initial size only decides how much memory is reserved up front. The default leaves
room for other processes on the same GPU; to reserve more (or less) memory up front,
set `initial_device_fraction` or `initial_device_bytes` in `MemoryPoolConfig` (see below).

If your system allows, you can redefine the memory pool sizes to better suit your use case.

### Memory Pool Configuration (Runtime)

You can also control the memory pool sizes directly at runtime when generating
a context. Sizes can be given as **percentages** (0.0-1.0 or 0-100) or as
**absolute bytes**. If you do not provide a config, the defaults above are used.

```cpp
heongpu::HEParams<heongpu::Scheme::CKKS> params;
params.multiplicative_depth = 3;
heongpu::HEContext<heongpu::Scheme::CKKS> context = params.make_context();

heongpu::MemoryPoolConfig pool_config;
pool_config.initial_device_fraction = 80.0f; // %80 (0.8f is also valid)
pool_config.max_device_fraction = 90.0f;     // %90
pool_config.initial_host_bytes = 256ULL * 1024 * 1024; // 256 MB
pool_config.max_host_fraction = 50.0f;                 // %50

context->generate(pool_config);
```

> Note: The memory pool is a process-wide singleton. The **first**
> initialization that enables the pool sets the pool sizes; subsequent contexts
> reuse the same pool. `use_memory_pool` is also process-wide: the most recently
> generated context decides whether new allocations use the pool, and buffers
> are always returned to the resource they were allocated from.

HEonGPU allocates from its own memory resource and does not change the process-wide RMM default, so other code in the process that uses RMM (for example RAPIDS libraries) keeps its own allocations. Set `pool_config.set_as_rmm_default = true` to make HEonGPU's pool the RMM default as well; the previous default is restored when a later context turns it off, and at exit.

`generate()` also raises the CUDA kernel stack size limit (`cudaLimitStackSize`) to the 2048 bytes HEonGPU's kernels need, if it is lower. Set `pool_config.raise_kernel_stack_size = false` to manage the limit yourself; `generate()` then throws if it is too low. `heongpu::initialize(config)` sets up the memory pool and the other process-wide state explicitly, and `heongpu::shutdown()` releases them once all HEonGPU objects are destroyed, so that they can be set up again with another configuration. See the documentation (Advanced Topics, Process-Wide State) for all process-wide state of HEonGPU.


## Storage Management

In HEonGPU, Fully Homomorphic Encryption (FHE) operations rely on an efficient storage management system that ensures data can seamlessly transition between host (CPU) and device (GPU) memory. The Storage Manager module enables this flexibility, allowing data to be stored and accessed as needed on either the host or device.

### How Storage Management Works

Objects can be stored in host or device memory, and the operations take them from either. The `ExecutionOptions` struct sets where the results go:

- `set_storage_type`: where the output of an operation is stored (`storage_type::device` or `storage_type::host`), and where the object an `*_inplace` operation changes ends up.

```c++
// ...
ExecutionOptions options;
options.set_storage_type(storage_type::device);

operators.add(input1, input2, output, options);
// ...
```

The inputs of an operation are taken by `const` reference and never change: an input in host memory is read from a temporary copy on the device and stays in host memory. To keep an input on the device for later operations, move it there once with `store_in_device()`. `set_initial_location` only applies to the object an `*_inplace` operation changes.

## Using HEonGPU in a downstream CMake project

Make sure HEonGPU is installed before integrating it into your project. The installed HEonGPU library provides a set of config files that make it easy to integrate HEonGPU into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(HEonGPU)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) HEonGPU::heongpu CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(<your-target> PROPERTIES POSITION_INDEPENDENT_CODE ON)
# ...
```

This applies to the default static library (`libheongpu.a`), which leaves the CUDA device link to your project.

HEonGPU can also be built as a shared library with `-D HEonGPU_BUILD_SHARED=ON` (`libheongpu.so`). The shared library does its own device link, so your project does not need to enable CUDA or separable compilation, and all libraries in a process that use it share one HEonGPU, including its memory pool:

```cmake
project(<your-project> LANGUAGES CXX)
find_package(HEonGPU)
target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu)
```

The shared library is versioned with the major and minor version (for example `libheongpu.so.1.4`), because its ABI can change between minor versions. Its kernels are internal and cannot be launched from outside the library.

## Versioning and Compatibility

HEonGPU follows [semantic versioning](https://semver.org/):
- **Major versions** (2.0, 3.0, ...) can change or remove APIs.
- **Minor versions** (1.4, 1.5, ...) add features without breaking code that compiles against the previous minor version. They can change the ABI, so rebuild code that uses HEonGPU; the shared library is versioned accordingly (`libheongpu.so.1.4`).
- **Patch versions** (1.4.1, ...) fix bugs without changing the API or the ABI.

APIs that will be removed are first marked `[[deprecated]]`, with the replacement in the warning, for at least one minor version, and removed in the next major version. HEonGPU 2.0 removed the APIs deprecated in 1.x; the documentation (Advanced Topics) lists them with their replacements.

Serialized data stays readable: HEonGPU 2.0 reads the data written by every 1.x version, and HEonGPU 1.3 and later read the data written by 2.0. HEonGPU 2.0 writes the format version 2 only; data of the format version 1, written by HEonGPU 1.2 and earlier, can still be loaded.

Supported toolchain: CUDA 12.8 or newer, a C++20 compiler (tested with GCC 11.4) and CMake 4.0 or newer.

### Thread safety
- A generated context is not changed afterwards and can be shared by threads.
- Keys and operators can be shared by threads that run operations on their own ciphertexts and plaintexts and their own CUDA streams, as the multi-stream examples do.
- Operations do not change their inputs, so threads can share them. An object that an operation writes, or that is moved between host and device memory, must not be used by another thread at the same time.
- An `HESession` and its copies can be used by several threads, each on its own CUDA stream with `session.on_stream(stream)`. A rotation by a step that has no key yet generates the key while the other rotations of the session wait.
- The memory pool is thread-safe. Call `heongpu::initialize()` and `heongpu::shutdown()` while no other thread uses HEonGPU.
- Key generation, encryption and multiparty computation can run from several threads. They draw from one process-wide random number generator. It takes one request at a time on the host but does not wait for the GPU, so the GPU work of requests on different CUDA streams runs concurrently. The numbers of a seed, as the common polynomials of multiparty computation, are drawn without other threads drawing in between and are the same on every GPU and every kind of CUDA stream.

## How to Cite HEonGPU

Please use the below BibTeX, to cite HEonGPU in academic papers.

```
@misc{cryptoeprint:2024/1543,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1543}
}
```

Please use the below BibTeX, to cite key-switching optimizations in academic papers.

Note: the key switching method of this paper is available only in release [`v1.1.3`](https://github.com/Alisah-Ozcan/HEonGPU/releases/tag/v1.1.3) and earlier.

```
@misc{cryptoeprint:2025/124,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{GPU} Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/124},
      year = {2025},
      url = {https://eprint.iacr.org/2025/124}
}
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)

<div align="center">

![enCRYPTON-Labs logo](./figures/encrypton_labs_amblem.png)

</div>
