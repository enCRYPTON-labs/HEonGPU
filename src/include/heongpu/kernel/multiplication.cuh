// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_MULTIPLICATION_H
#define HEONGPU_MULTIPLICATION_H

#include "cuda_runtime.h"
#include "gpuntt/common/modular_arith.cuh"
#include <heongpu/kernel/defines.h>

namespace heongpu
{
    // Homomorphic Multiplication Kernels

    __global__ void cross_multiplication(Data64* in1, Data64* in2, Data64* out,
                                         Modulus64* modulus, int n_power,
                                         int decomp_size);

    __global__ void
    fast_convertion(Data64* in1, Data64* in2, Data64* out1, Modulus64* ibase,
                    Modulus64* obase, Modulus64 m_tilde,
                    Data64 inv_prod_q_mod_m_tilde, Data64* inv_m_tilde_mod_Bsk,
                    Data64* prod_q_mod_Bsk, Data64* base_change_matrix_Bsk,
                    Data64* base_change_matrix_m_tilde,
                    Data64* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size);

    __global__ void fast_floor(
        Data64* in_baseq_Bsk, Data64* out1, Modulus64* ibase, Modulus64* obase,
        Modulus64 plain_modulus, Data64* inv_punctured_prod_mod_base_array,
        Data64* base_change_matrix_Bsk, Data64* inv_prod_q_mod_Bsk,
        Data64* inv_punctured_prod_mod_B_array, Data64* base_change_matrix_q,
        Data64* base_change_matrix_msk, Data64 inv_prod_B_mod_m_sk,
        Data64* prod_B_mod_q, int n_power, int ibase_size, int obase_size);

    __global__ void threshold_kernel(Data64* plain_in, Data64* output,
                                     Modulus64* modulus,
                                     Data64* plain_upper_half_increment,
                                     Data64 plain_upper_half_threshold,
                                     int n_power, int decomp_size);

    __global__ void cipherplain_kernel(Data64* cipher, Data64* plain_in,
                                       Data64* output, Modulus64* modulus,
                                       int n_power, int decomp_size);

    __global__ void cipherplain_multiplication_kernel(Data64* in1, Data64* in2,
                                                      Data64* out,
                                                      Modulus64* modulus,
                                                      int n_power);

    __global__ void
    cipher_constant_plain_multiplication_kernel(Data64* in1, double in2,
                                                Data64* out, Modulus64* modulus,
                                                double two_pow_64, int n_power);

    __global__ void cipherplain_multiply_accumulate_indexed_kernel(
        Data64* baby_results, Data64* plaintexts, Data64* out,
        Modulus64* modulus, int* ct_indices, int iteration_count,
        int current_decomp_count, int first_decomp_count, int n_power);

    __global__ void cipher_div_by_i_kernel(Data64* in1, Data64* out,
                                           Data64* ntt_table,
                                           Modulus64* modulus, int n_power);

    __global__ void cipher_mult_by_i_kernel(Data64* in1, Data64* out,
                                            Data64* ntt_table,
                                            Modulus64* modulus, int n_power);

    __global__ void cipher_mult_by_gaussian_integer_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* out,
        Data64* ntt_table, Modulus64* modulus, int n_power);

    __global__ void cipher_add_by_gaussian_integer_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* out,
        Data64* ntt_table, Modulus64* modulus, int n_power);

    __global__ void cipher_mult_by_gaussian_integer_and_add_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* accumulator,
        Data64* ntt_table, Modulus64* modulus, int n_power);

} // namespace heongpu
#endif // HEONGPU_MULTIPLICATION_H
