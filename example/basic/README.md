# HEonGPU Basic Examples

The basic example directory in the HEonGPU repository contains sample programs that demonstrate how to utilize various features of the HEonGPU library. These examples are designed to help users understand and experiment with the library's functionalities, from basic encryption and decryption to more advanced homomorphic operations.

## Overview

- [1_basic_bfv.cpp](1_basic_bfv.cpp): Demonstrates basic arithmetic operations on encrypted integers using the BFV scheme, including encryption, evaluation, and decryption.

- [2_basic_ckks.cpp](2_basic_ckks.cpp): Introduces the CKKS scheme for approximate arithmetic on real and complex numbers, showing encoding, encryption, evaluation, and decryption.

- [3_basic_memorypool_config.cpp](3_basic_memorypool_config.cpp): Shows how to configure the memory pool (host/device size via percentages or bytes) before generating a context->

- [4_bfv_key_switching.cpp](4_bfv_key_switching.cpp): Rotations and key switching in the BFV scheme: it rotates rows and columns with Galois keys and switches a ciphertext to another secret key.

- [5_ckks_key_switching.cpp](5_ckks_key_switching.cpp): Rotations and key switching in the CKKS scheme: it rotates the slots with Galois keys and switches a ciphertext to another secret key.

- [6_ckks_coefficient_encoding.cpp](6_ckks_coefficient_encoding.cpp): Demonstrates CKKS coefficient encoding (`encoding::coefficient`) end-to-end, including encode/encrypt/decrypt/decode and encoding-mismatch validation.

- [7_ckks_coeff_to_slot_roundtrip.cpp](7_ckks_coeff_to_slot_roundtrip.cpp): Demonstrates dedicated `generate_encoding_transform_context(...)` API with explicit `CtoS_start_level` / `StoC_start_level` inputs (where `StoC_start_level` is the input level of `slot_to_coeff`), then passes that context into public `coeff_to_slot` / `slot_to_coeff` APIs and validates roundtrip correctness.

- [8_default_stream_usage.cpp](8_default_stream_usage.cpp): Shows how to perform encrypted computations using the default CUDA stream, highlighting the integration of GPU processing in homomorphic encryption.

- [9_multi_stream_usage_way1.cpp](9_multi_stream_usage_way1.cpp): Uses OpenMP to assign a CUDA stream to each CPU thread, demonstrating a multi-stream approach for executing parallel encrypted operations efficiently.

- [10_multi_stream_usage_way2.cpp](10_multi_stream_usage_way2.cpp): Implements a basic encrypted operation using multiple CUDA streams, where each stream is invoked by a single CPU thread. This example highlights an alternative method to leverage multi-stream functionality for concurrent GPU processing.

- [11_basic_bfv_logic.cpp](11_basic_bfv_logic.cpp): Demonstrates basic logic operations on encrypted binarys using the BFV scheme, including encryption, evaluation, and decryption.

- [12_basic_ckks_logic.cpp](12_basic_ckks_logic.cpp): Demonstrates basic logic operations on encrypted binarys using the CKKS scheme, including encryption, evaluation, and decryption.

- [13_bfv_serialization.cpp](13_bfv_serialization.cpp): Demonstrates serialization and deserialization workflows for BFV context and keys.

- [14_ckks_serialization.cpp](14_ckks_serialization.cpp): Demonstrates serialization and deserialization workflows for CKKS context and keys.

- [15_basic_tfhe.cpp](15_basic_tfhe.cpp): Introduces the TFHE scheme basics and shows simple Boolean gate operations.
