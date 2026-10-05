// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BOOTSTRAPPING_H
#define HEONGPU_BOOTSTRAPPING_H

#include <curand_kernel.h>
#include "gpuntt/common/modular_arith.cuh"
#include "gpufft/complex.cuh"
#include <heongpu/kernel/small_ntt.cuh>

namespace heongpu
{

    template <typename T>
    __device__ T rotated_access(T* data, int& rotate, int& idx, int& n_power)
    {
        int n = 1 << n_power;
        int new_location = idx + rotate;

        if (new_location < 0)
        {
            new_location = new_location + n;
        }
        else
        {
            int mask = n - 1;
            new_location = new_location & mask;
        }

        return data[new_location];
    }

    __device__ int exponent_calculation(int& index, int& n);

    __device__ int matrix_location(int& index);

    __device__ int matrix_reverse_location(int& index);

    __global__ void E_diagonal_generate_kernel(Complex64* output, int n_power);

    __global__ void E_diagonal_inverse_generate_kernel(Complex64* output,
                                                       int n_power);

    __global__ void E_diagonal_inverse_matrix_mult_single_kernel(
        Complex64* input, Complex64* output, bool last, int n_power);

    __global__ void E_diagonal_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first1, bool first2, int n_power);

    __global__ void E_diagonal_inverse_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first, bool last, int n_power);

    __global__ void complex_vector_scale_kernel(Complex64* data,
                                                Complex64 scale, int n_power);

    __global__ void vector_rotate_kernel(Complex64* input, Complex64* output,
                                         int rotate_index, int n_power);

    // TODO: work it for multiple RNS prime (currently it only works for single
    // prime)
    __global__ void mod_raise_kernel(Data64* input, Data64* output,
                                     Modulus64* modulus, int n_power);

    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////

    // Modulus should be power of 2.
    __device__ int32_t torus_modulus_switch_log(int32_t& input,
                                                int& modulus_log);

    __global__ void
    tfhe_nand_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                              int32_t* input1_a, int32_t* input1_b,
                              int32_t* input2_a, int32_t* input2_b,
                              int32_t encoded, int n);

    __global__ void
    tfhe_and_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n);

    __global__ void
    tfhe_and_first_not_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                                       int32_t* input1_a, int32_t* input1_b,
                                       int32_t* input2_a, int32_t* input2_b,
                                       int32_t encoded, int n);

    __global__ void
    tfhe_nor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n);

    __global__ void
    tfhe_or_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                            int32_t* input1_a, int32_t* input1_b,
                            int32_t* input2_a, int32_t* input2_b,
                            int32_t encoded, int n);

    __global__ void
    tfhe_xnor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                              int32_t* input1_a, int32_t* input1_b,
                              int32_t* input2_a, int32_t* input2_b,
                              int32_t encoded, int n);

    __global__ void
    tfhe_xor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n);

    __global__ void tfhe_not_comp_kernel(int32_t* output_a, int32_t* output_b,
                                         int32_t* input1_a, int32_t* input1_b,
                                         int n);

    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////

    struct int32_t2
    {
        int32_t value[2];

        __device__ int32_t2() : value{0, 0} {}
    };

    struct uint64_t2
    {
        Data64 value[2];

        __device__ uint64_t2() : value{0ULL, 0ULL} {}
    };

    __global__ void tfhe_bootstrapping_kernel(
        const int32_t* input_a, const int32_t* input_b, int32_t* output,
        const Data64* boot_key,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus,
        const int32_t encoded, const int32_t bk_offset, const int32_t bk_mask,
        const int32_t bk_half, int n, int N, int N_power, int k, int bk_bit,
        int bk_length);
    /*
        __global__ void tfhe_bootstrapping_kernel_unique_step1
        (
            const int32_t* input_a, const int32_t* input_b, Data64* output,
       const Data64* boot_key, const Root64* __restrict__
       forward_root_of_unity_table, const Modulus64 modulus, const int32_t
       encoded, const int32_t bk_offset, const int32_t bk_mask, const int32_t
       bk_half, int n, int N, int N_power, int k, int bk_bit, int bk_length);
    */

    __global__ void tfhe_bootstrapping_kernel_unique_step1(
        const int32_t* input_a, const int32_t* input_b, Data64* output,
        const Data64* boot_key,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Modulus64 modulus, const int32_t encoded, const int32_t bk_offset,
        const int32_t bk_mask, const int32_t bk_half, int n, int N, int N_power,
        int k, int bk_bit, int bk_length);

    __global__ void tfhe_bootstrapping_kernel_regular_step1(
        const int32_t* input_a, const int32_t* input_b, const int32_t* input_c,
        Data64* output, const Data64* boot_key, int boot_index,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Modulus64 modulus, const int32_t bk_offset, const int32_t bk_mask,
        const int32_t bk_half, int n, int N, int N_power, int k, int bk_bit,
        int bk_length);

    __global__ void tfhe_bootstrapping_kernel_unique_step2(
        const Data64* input, const int32_t* input_b, int32_t* output,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus,
        const int32_t encoded, int n, int N, int N_power, int k, int bk_length);

    __global__ void tfhe_bootstrapping_kernel_regular_step2(
        const Data64* input, int32_t* output,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus, int n, int N,
        int k, int bk_length);

    __global__ void tfhe_sample_extraction_kernel(const int32_t* input,
                                                  int32_t* output_a,
                                                  int32_t* output_b, int N,
                                                  int k, int index);

    __global__ void tfhe_key_switching_kernel(
        const int32_t* input_a, const int32_t* input_b, int32_t* output_a,
        int32_t* output_b, const int32_t* ks_key_a, const int32_t* ks_key_b,
        int ks_base_bit_, int ks_length_, int n, int N, int k);

} // namespace heongpu

#endif // HEONGPU_BOOTSTRAPPING_H
