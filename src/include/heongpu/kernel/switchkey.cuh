// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SWITCHKEY_H
#define HEONGPU_SWITCHKEY_H

#include "cuda_runtime.h"
#include "gpuntt/common/modular_arith.cuh"

namespace heongpu
{
    __global__ void keyswitch_multiply_accumulate_kernel(
        Data64* input, const Data64* __restrict__ relinkey, Data64* output,
        Modulus64* modulus, int n_power, int Q_tilda_size, int iteration_count1,
        int iteration_count2);

    __global__ void keyswitch_multiply_accumulate_leveled_method_II_kernel(
        Data64* input, const Data64* __restrict__ relinkey, Data64* output,
        Modulus64* modulus, int first_rns_mod_count,
        int current_decomp_mod_count, int current_rns_mod_count,
        int iteration_count1, int iteration_count2, int level, int n_power);

    __global__ void
    divide_round_lastq_kernel(Data64* input, Data64* ct, Data64* output,
                              Modulus64* modulus, Data64* half,
                              Data64* half_mod, Data64* last_q_modinv,
                              int n_power, int decomp_mod_count);

    __global__ void divide_round_lastq_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int decomp_mod_count);

    __global__ void divide_round_lastq_leveled_stage_one_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, int n_power, int first_decomp_count,
        int current_decomp_count);

    __global__ void divide_round_lastq_leveled_stage_two_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count);

    __global__ void divide_round_lastq_leveled_stage_two_switchkey_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count);

    __global__ void move_cipher_leveled_kernel(Data64* input, Data64* output,
                                               int n_power,
                                               int current_decomp_count);

    __global__ void divide_round_lastq_rescale_kernel(
        Data64* input_last, Data64* input, Data64* output, Modulus64* modulus,
        Data64* last_q_modinv, int n_power, int current_decomp_count);

    __global__ void base_conversion_DtoB_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int l, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_DtoQtilde_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int l, int Q_tilda, int d);

    __global__ void base_conversion_DtoB_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int d_tilda, int d, int r_prime, int* mod_index);

    __global__ void base_conversion_DtoQtilde_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int d, int current_Qtilda_size, int current_Q_size, int level,
        int* mod_index);

    // base_conversion_DtoQtilde_relin_leveled_kernel for digits of one prime:
    // lifts each coefficient to its centered representative and reduces it
    // by the other moduli (Q primes below current_Q_size, then the P primes,
    // which start level positions later).
    __global__ void single_prime_digits_to_Qtilde_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus, int n_power,
        int current_Qtilda_size, int current_Q_size, int level);

    __global__ void multiply_accumulate_extended_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* B_prime,
        int n_power, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_BtoD_relin_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_BtoD_relin_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index);

    __global__ void divide_round_lastq_extended_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size);

    __global__ void divide_round_lastq_extended_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size);

    __global__ void divide_round_lastq_extended_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, Data64* last_q_modinv, int n_power, int Q_prime_size,
        int Q_size, int first_Q_prime_size, int first_Q_size, int P_size);

    // TODO: Find efficient way!
    __global__ void global_memory_replace_kernel(Data64* input, Data64* output,
                                                 int n_power);

    __global__ void
    global_memory_replace_offset_kernel(Data64* input, Data64* output,
                                        int current_decomposition_count,
                                        int n_power);

    __global__ void addition_switchkey(Data64* in1, Data64* in2, Data64* out,
                                       Modulus64* modulus, int n_power);

    __global__ void negacyclic_shift_poly_coeffmod_kernel(Data64* cipher_in,
                                                          Data64* cipher_out,
                                                          Modulus64* modulus,
                                                          int shift,
                                                          int n_power);

    ////////////////////////////////////////
    // Double Hoisting Kernels

    // NTT-domain Galois automorphism: permutes NTT slots directly,
    // avoiding the INTT->coeff-permute->NTT round trip.
    // perm_table: precomputed gather indices of size N, where
    //   perm(j) = br((galois_elt * (2*br(j)+1) % 2N - 1) / 2)
    // Grid: dim3((n >> 8), pql_count, 2), 256
    __global__ void galois_permute_ntt_pql_kernel(Data64* input, Data64* output,
                                                  int galois_elt, int n_power,
                                                  int pql_count);

    __global__ void broadcast_scale_P_kernel(Data64* c_ntt, Data64* output,
                                             Data64* P_mod_q,
                                             Modulus64* pq_modulus, int n_power,
                                             int current_decomp_count,
                                             int pql_count);

    __global__ void addition_pql_kernel(Data64* in1, Data64* in2, Data64* out,
                                        Modulus64* pq_modulus, int n_power,
                                        int pql_count);

    ////////////////////////////////////////
    // Optimized Hoisting-Rotations

    __global__ void divide_round_lastq_permute_ckks_kernel(
        Data64* input, Data64* input2, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int first_Q_prime_size,
        int first_Q_size, int P_size);

    __global__ void divide_round_lastq_permute_bfv_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int P_size);

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

} // namespace heongpu
#endif // HEONGPU_SWITCHKEY_H
