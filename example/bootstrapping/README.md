# Bootstrapping in HEonGPU

Homomorphic encryption allows computations to be performed on encrypted data without decryption. However, as computations progress, the noise in the ciphertext grows, eventually making further operations impossible. Bootstrapping is the process of "refreshing" a ciphertext to reduce this noise, enabling deeper computations and extending the lifespan of the ciphertext.

Currently, HEonGPU supports `4` different bootstrapping types for the `CKKS` scheme. In the near future, support for the `TFHE` scheme will be added, including `TFHE` bootstrapping to enable efficient bit-level encrypted computations.

## CKKS Bootstrapping

The `CKKS` bootstrapping implementation in HEonGPU demonstrates the initialization and execution of the bootstrapping process entirely on the GPU.

#### [CKKS Regular Bootstrapping](1_ckks_regular_bootstrapping.cpp)

<div align="center">

```mermaid
graph LR
  A["m(x)"] -->|ModRaise| B["m(x) + qI(x)"]
  B["m(x) + qI(x)"] -->|CoeffToSlot| C["Encode(<b>m</b> + q<b>I</b>)"]
  C["Encode(<b>m</b> + q<b>I</b>)"] -->|EvalMod| D["Encode(<b>m</b>)"]
  D["Encode(<b>m</b>)"] -->|SlotToCoeff| E["m(x)"]
```
</div>

In cases where the message is a complex number, Regular Bootstrapping is applied. 

<div align="center">
m(x) = Encode(<b>m</b>) where <b>m</b> &in; &#8450 
</div>

The [1_ckks_regular_bootstrapping.cpp](1_ckks_regular_bootstrapping.cpp) provides an example of `Regular` bootstrapping with a polynomial degree of 65536, a dense secret key (h=192) and a temporary sparse secret key (h=32) that is only used around `ModRaise` (dense-to-sparse and sparse-to-dense switch keys).

#### [CKKS Slim Bootstrapping](2_ckks_slim_bootstrapping.cpp)

<div align="center">

```mermaid
graph LR
  A["Encode(<b>z</b>)"] -->|SlotToCoeff| B["z(x)"]
  B["z(x)"] -->|ModRaise| C["z(x) + qI(x)"]
  C["z(x) + qI(x)"] -->|CoeffToSlot| D["Encode(<b>z</b> + q<b>I</b>)"]
  D["Encode(<b>z</b> + q<b>I</b>)"] -->|EvalMod| E["Encode(<b>z</b>)"]
```
</div>

Unlike `Regular` Bootstrapping, `Slim` Bootstrapping is designed for messages in the real number domain. The bootstrapping process begins with `Slot To Coeff` (StoC) rather than ModRaise, ensuring that EvalMod is applied exclusively to the real part of the message. For a comprehensive explanation and technical details, please refer to the papers listed below.

<div align="center">
m(x) = Encode(<b>m</b>) where <b>m</b> &in; &reals;
</div>

The [2_ckks_slim_bootstrapping.cpp](2_ckks_slim_bootstrapping.cpp) provides an example of `Slim` bootstrapping with a polynomial degree of 65536. Since slot values become coefficients directly after `SlotToCoeff`, `Slim` bootstrapping uses a larger `message_ratio` (1024) than `Regular` bootstrapping.

#### [CKKS Bit Bootstrapping](3_ckks_bit_bootstrapping.cpp)

CKKS `Bit` Bootstrapping, like Slim Bootstrapping, starts with `Slot To Coeff` (StoC). Unlike `Slim` Bootstrapping, message that will be applied `Bit` Bootstrapping, is in the binary domain. As the message is binary, Eval<sub>f<sub>Binboot</sub></sub> is applied instead of EvalMod. The bits are placed at q/2 before `ModRaise`, so the ciphertext scale (&Delta;) does not need to be related to the last modulus (q). For a comprehensive explanation and technical details, please refer to the papers listed below.

<div align="center">
m(x) = Encode(<b>m</b>) where <b>m</b> &in; &#120121;
</div>

The [3_ckks_bit_bootstrapping.cpp](3_ckks_bit_bootstrapping.cpp) provides an example of `Bit` bootstrapping with a polynomial degree of 65536.

#### [CKKS Gate Bootstrapping](4_ckks_gate_bootstrapping.cpp)

CKKS `Gate` Bootstrapping is quite similar to `Bit` Bootstrapping, but instead of Eval<sub>f<sub>Binboot</sub></sub>, it uses Eval<sub>f<sub>Gateboot</sub></sub>. The sum of the two input bits is placed at q/3 before `ModRaise`, so, as in `Bit` Bootstrapping, the ciphertext scale (&Delta;) does not need to be related to the last modulus (q). `Gate` Bootstrapping applies the logic operation directly during the bootstrapping process, similar to how it is done in `TFHE`. For a comprehensive explanation and technical details, please refer to the papers listed below.

<div align="center">
m(x) = Encode(<b>m</b>) where <b>m</b> &in; &#120121;
</div>

The [4_ckks_gate_bootstrapping.cpp](4_ckks_gate_bootstrapping.cpp) provides an example of `Gate` bootstrapping with a polynomial degree of 65536.

All logic gate can be applied with CKKS `Gate` Bootstrapping:
- AND_bootstrapping - NAND_bootstrapping
- OR_bootstrapping - NOR_bootstrapping
- XOR_bootstrapping - XNOR_bootstrapping

### Implementation Details

`CKKS` bootstrapping primarily consists of four main stages: `Mod Raise`, `Coeff to Slot`, `Approximate Modular Reduction`, and `Slot to Coeff`. Among these, `Coeff to Slot` is the most time-consuming part of the bootstrapping. `Coeff to Slot` and `Slot to Coeff` stages primarily involve matrix multiplication, which can be accelerated using the BSGS method. However, HEonGPU employs the most optimized approach described in the [Faster Homomorphic Discrete Fourier Transforms and Improved FHE Bootstrapping](https://eprint.iacr.org/2018/1073.pdf) paper, significantly minimizing both the number of multiplications and rotations. Moreover, HEonGPU does not use sparse packing as described in the [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153.pdf) paper; instead, it utilizes all slots of CKKS.

Homomorphic operation number and depth consume for homomorphic DFT(from [paper](https://eprint.iacr.org/2018/1073.pdf)):

<div align="center">

|                  | Naive       | BSGS        | Ours with radix r   | Hybrid with radix r   |
|:----------------:|:-----------:|:-----------:|:-------------------:|:---------------------:|
| # Hadamard Mult  | O(n)    | O(n)   | O(r log<sub>r</sub> n)  | O(r log<sub>r</sub> n)    |
| # Slot Shifting  | O(n)   | O(&radic;n)| O(r log<sub>r</sub> n)| O(&radic;r log<sub>r</sub> n)|
| Depth            | 1           | 1           | O(log<sub>r</sub> n)    | O(log<sub>r</sub> n)      |

</div>
Although this method results in higher depth consumption in stages `Coeff to Slot` and `Slot to Coeff` compared to Naive and BSGS approaches, it provides a significant performance improvement. The `EncodingMatrixConfig` of each stage sets its start level, its level loss and its baby-step/giant-step ratio. The matrix multiplications use double hoisting and require `KEYSWITCHING_METHOD_II` (more than one P prime). There is also an option to store all galois keys on the CPU, but this would significantly slow down the bootstrapping process. However, the key size still considerably large, so it is important to keep this in mind while perfoming bootstrapping!

`Approximate Modular Reduction` evaluates a Chebyshev interpolation of the cosine followed by double-angle steps. The `EvalModConfig` class configures it (`K`, polynomial degree, number of double-angle steps, `message_ratio`).

The vandermonde matrix is automatically generated on GPU based on the `BootstrappingConfig` class and stored on the GPU in a ready-to-use and encoded format for processing.

> **⚠ Warning:** Bootstrapping was tested on a system with 128 GB RAM and an NVIDIA RTX 4090(24 GB). If you are working with a system with lower specifications, please refer to [define.h](../../src/include/heongpu/kernel/defines.h) and adjust the memory pool settings according to your system.


### The papers used while implementing HEonGPU CKKS bootstrapping

- [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153.pdf) ✓
- [Improved Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/1043.pdf) ✓
- [Faster Homomorphic Discrete Fourier Transforms and Improved FHE Bootstrapping](https://eprint.iacr.org/2018/1073.pdf) ✓
- [Better Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2019/688.pdf) ✓
- [Efficient Bootstrapping for Approximate Homomorphic Encryption with Non-Sparse Keys](https://eprint.iacr.org/2020/1203.pdf) ✓
- [Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs](https://eprint.iacr.org/2021/508.pdf) ✓
- [BLEACH: Cleaning Errors in Discrete Computations over CKKS](https://eprint.iacr.org/2022/1298.pdf) ✓
- [Bootstrapping Bits with CKKS](https://eprint.iacr.org/2024/767.pdf) ✓
- [Bootstrapping Small Integers With CKKS](https://eprint.iacr.org/2024/1637.pdf) (added soon)
- [Modular Reduction in CKKS](https://eprint.iacr.org/2024/1638.pdf) (added soon)



## How to Cite HEonGPU CKKS Bootstrapping

Please keep using the below BibTeX to cite HEonGPU in academic papers. We are currently working on a new paper specifically for GPU implementation of CKKS Bootstrapping, and until its release, please continue to use this citation.

```
@misc{cryptoeprint:2024/1543,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1543}
}
```

## Acknowledgments

I would like to appreciate `Efe İzbudak` for his mathematical support and the great teamwork during all `CKKS` bootstrapping implementations.

#

If you have further questions or require additional information, feel free to reach out at:
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)
