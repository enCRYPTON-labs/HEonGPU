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


### 🚨 **New Scheme: [TFHE (Torus Fully Homomorphic Encryption)](example/basic/13_basic_tfhe.cpp)**

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
  </tr>
  <tr>
    <td rowspan="2"><b>Regular Bootstrapping</b></td>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>9 Level</td>
    <td>51.02 ms</td>
    <td>3.11 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>109.28 ms</td>
    <td>3.33 µs</td>
  </tr>
  <tr>
    <td rowspan="4"><b>Slim Bootstrapping</b></td>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>0 Level</td>
    <td>17.56 ms</td>
    <td>1.07 µs</td>
  </tr>
  <tr>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>9 Level</td>
    <td>30.87 ms</td>
    <td>1.88 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>38.55 ms</td>
    <td>1.18 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>9 Level</td>
    <td>70.59 ms</td>
    <td>2.15 µs</td>
  </tr>
  <tr>
    <td rowspan="4"><b>Bit Bootstrapping</b></td>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>0 Level</td>
    <td>22.53 ms</td>
    <td>1.38 µs</td>
  </tr>
  <tr>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>8 Level</td>
    <td>37.51 ms</td>
    <td>2.29 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>50.72 ms</td>
    <td>1.55 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>8 Level</td>
    <td>82.43 ms</td>
    <td>2.52 µs</td>
  </tr>
  <tr>
    <td rowspan="4"><b>Gate Bootstrapping*</b></td>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>0 Level</td>
    <td>29.96–31.85 ms</td>
    <td>1.83–1.94 µs</td>
  </tr>
  <tr>
    <td>2<sup>15</sup></td>
    <td>2<sup>14</sup></td>
    <td>7 Level</td>
    <td>45.36–47.50 ms</td>
    <td>2.77–2.90 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>0 Level</td>
    <td>62.90–66.55 ms</td>
    <td>1.92–2.03 µs</td>
  </tr>
  <tr>
    <td>2<sup>16</sup></td>
    <td>2<sup>15</sup></td>
    <td>7 Level</td>
    <td>99.45–102.17 ms</td>
    <td>3.04–3.12 µs</td>
  </tr>
</table>
</div>
Dense secret key (h=192) with a temporary sparse secret key (h=32), median of 5 runs. Remaining Level is the number of levels left before the next bootstrapping.

*: For all gates



### 🚨 **New Feature: [CKKS Regular Bootstrapping](example/bootstrapping/README.md)**

HEonGPU now includes support for `CKKS Regular` Bootstrapping, enabling efficient evaluation of deep computational circuits with high precision and security. On an NVIDIA RTX 4090, it performs `CKKS` Regular Bootstrapping for 
N=65536 in about 110 ms.

### 🚨 **New Feature: [Multiparty Computation (MPC) Support](example/mpc/README.md)**

HEonGPU now includes support for **Multiparty Computation (MPC)** protocols, providing a secure and collaborative framework for encrypted computations. By incorporating *Multiparty Homomorphic Encryption (MHE)* capabilities, the library enables distributed computations with threshold encryption models such as `N-out-of-N`. The implementation is fully optimized for GPU environments, delivering minimal latency and maximum performance in collaborative settings.

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.30.4
- [GCC](https://gcc.gnu.org/)
- [GMP](https://gmplib.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4
- [OpenSSL](https://www.openssl.org/) >= 1.1.0
- [ZLIB](https://zlib.net/)

### Third-Party Dependencies
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
- [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT)
- [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU)
- [RMM](https://github.com/rapidsai/rmm)
- [GoogleTest](https://github.com/google/googletest)

HEonGPU automatically handle third-party dependencies like GPU-NTT, GPU-FFT, RMM, GoogleTest.
Default CUDA architectures: `80;86;89;90`. Override with `-D CMAKE_CUDA_ARCHITECTURES=<...>` if needed.

### Build & Install

To build and install HEonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

<div align="center">

| GPU Architecture | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value) |
|:----------------:|:---------------------------------------------------:|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada	 | 89, 90 |

</div>

```bash
$ cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
$ cmake --build ./build/
$ sudo cmake --install build
```
Available build types: `Debug`, `Release` (default), `RelWithDebInfo`, `MinSizeRel`. Override with `-D CMAKE_BUILD_TYPE=<type>`. These propagate to all bundled libraries (GPU-FFT, GPU-NTT, RNGonGPU).

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

```c++
#include <heongpu/heongpu.hpp>

int main() {   
    heongpu::HEContext<heongpu::Scheme::BFV> context =
        heongpu::GenHEContext<heongpu::Scheme::BFV>();

    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_default_values(1);
    int plain_modulus = 1032193;
    context.set_plain_modulus(plain_modulus);
    context.generate();

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

- **Galois Key Capability:** `MAX_SHIFT` (8) controls the maximum rotation capability for Galois keys. __Don't forget to change it if you need more rotation steps(for default galois key generation)__.

- **Device**:
  - Initial size (`0.9`): 90% of available GPU memory.
  - Max size (`0.95`): 95% of available GPU memory.
- **Host**:
  - Initial size (default): 100 MB when no runtime config is provided.
  - Max size (`0.4`): 40% of available CPU memory.

If your system allows, you can redefine the memory pool sizes to better suit your use case.

### Memory Pool Configuration (Runtime)

You can also control the memory pool sizes directly at runtime when generating
a context. Sizes can be given as **percentages** (0.0-1.0 or 0-100) or as
**absolute bytes**. If you do not provide a config, the defaults above are used.

```cpp
heongpu::HEContext<heongpu::Scheme::CKKS> context =
    heongpu::GenHEContext<heongpu::Scheme::CKKS>();

context.set_poly_modulus_degree(8192);
context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});

heongpu::MemoryPoolConfig pool_config;
pool_config.initial_device_fraction = 80.0f; // %80 (0.8f is also valid)
pool_config.max_device_fraction = 90.0f;     // %90
pool_config.initial_host_bytes = 256ULL * 1024 * 1024; // 256 MB
pool_config.max_host_fraction = 50.0f;                 // %50

context.generate(pool_config);
```

> Note: The memory pool is a process-wide singleton. The **first**
> initialization sets the pool sizes; subsequent contexts reuse the same pool.


## Storage Management

In HEonGPU, Fully Homomorphic Encryption (FHE) operations rely on an efficient storage management system that ensures data can seamlessly transition between host (CPU) and device (GPU) memory. The Storage Manager module enables this flexibility, allowing data to be stored and accessed as needed on either the host or device.

### How Storage Management Works

The `ExecutionOptions` struct in HEonGPU determines how input data is handled during and after GPU computations. Specifically:

- `set_storage_type`: Defines where the output data should reside after the operation (`DEVICE` or `HOST`).

- `set_initial_location`: Determines whether the input data should return to its original location after the computation.

```c++
// ...
ExecutionOptions options;
options.set_storage_type(storage_type::DEVICE)
       .set_initial_location(true);

operators.add(input1, input2, output, options);
// ...
```

- Simple case:
  - If `input1` and `input2` are initially on the `HOST`, they will be copied to the `DEVICE` for the computation.
  - After the operation, since `set_initial_location(true)` is specified, both `input1` and `input2` will return to the `HOST`.
  - The `output` will remain on the `DEVICE`, as specified by `set_storage_type(storage_type::DEVICE)`.

- All case:

<div align="center">

| `set_storage_type` | `set_initial_location` | Input Locations (Before)            | Input Locations (After)             | Output Location After Computation |
|:------------------:|:----------------------:|:-----------------------------------:|:-----------------------------------:|:---------------------------------:|
| `DEVICE`           | `true`                | `input1: HOST, input2: HOST`        | `input1: HOST, input2: HOST`        | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: HOST, input2: HOST`        | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `true`                | `input1: HOST, input2: DEVICE`      | `input1: HOST, input2: DEVICE`      | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: HOST, input2: DEVICE`      | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `true`                | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `HOST`             | `true`                | `input1: HOST, input2: HOST`        | `input1: HOST, input2: HOST`        | `HOST`                               |
| `HOST`             | `false`               | `input1: HOST, input2: HOST`        | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `true`                | `input1: HOST, input2: DEVICE`      | `input1: HOST, input2: HOST`        | `HOST`                               |
| `HOST`             | `false`               | `input1: HOST, input2: DEVICE`      | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `true`                | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `false`               | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |

</div>

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

Note: `heongpu::keyswitching_type::KEYSWITCHING_METHOD_III` is available only in release [`v1.1.3`](https://github.com/Alisah-Ozcan/HEonGPU/releases/tag/v1.1.3) and earlier.

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
