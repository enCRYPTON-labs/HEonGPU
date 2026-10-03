// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/multiplication.cuh>

namespace heongpu
{
    __global__ void
    fast_convertion(Data64* in1, Data64* in2, Data64* out1, Modulus64* ibase,
                    Modulus64* obase, Modulus64 m_tilde,
                    Data64 inv_prod_q_mod_m_tilde, Data64* inv_m_tilde_mod_Bsk,
                    Data64* prod_q_mod_Bsk, Data64* base_change_matrix_Bsk,
                    Data64* base_change_matrix_m_tilde,
                    Data64* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // cipher count * 2 // for input

        int location = idx + (((idy % 2) * ibase_size)
                              << n_power); // ibase_size = decomp_modulus_count
        Data64* input = ((idy >> 1) == 0) ? in1 : in2;

        Data64 temp[MAX_BSK_SIZE];
        Data64 temp_[MAX_BSK_SIZE];

        // taking input from global and mult with m_tilde
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp_[i] = input[location + (i << n_power)];
            temp[i] = OPERATOR_GPU_64::mult(temp_[i], m_tilde.value, ibase[i]);
            temp[i] = OPERATOR_GPU_64::mult(
                temp[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        // for Bsk
        Data64 temp2[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp2[i] = 0;
#pragma unroll
            for (int j = 0; j < ibase_size; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    temp[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp2[i] = OPERATOR_GPU_64::add(temp2[i], mult, obase[i]);
            }
        }

        // for m_tilde
        temp2[obase_size] = 0;
#pragma unroll
        for (int j = 0; j < ibase_size; j++)
        {
            Data64 temp_in = OPERATOR_GPU_64::reduce_forced(temp[j], m_tilde);
            Data64 mult = OPERATOR_GPU_64::mult(
                temp_in, base_change_matrix_m_tilde[j], m_tilde);
            temp2[obase_size] =
                OPERATOR_GPU_64::add(temp2[obase_size], mult, m_tilde);
        }

        // sm_mrq
        Data64 m_tilde_div_2 = m_tilde.value >> 1;
        Data64 r_m_tilde = OPERATOR_GPU_64::mult(
            temp2[obase_size], inv_prod_q_mod_m_tilde, m_tilde);
        r_m_tilde = m_tilde.value - r_m_tilde;

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data64 temp3 = r_m_tilde;
            if (temp3 >= m_tilde_div_2)
            {
                temp3 = obase[i].value - m_tilde.value;
                temp3 = OPERATOR_GPU_64::add(temp3, r_m_tilde, obase[i]);
            }

            temp3 = OPERATOR_GPU_64::mult(temp3, prod_q_mod_Bsk[i], obase[i]);
            temp3 = OPERATOR_GPU_64::add(temp2[i], temp3, obase[i]);
            temp2[i] =
                OPERATOR_GPU_64::mult(temp3, inv_m_tilde_mod_Bsk[i], obase[i]);
        }

        int location2 = idx + ((idy * (obase_size + ibase_size)) << n_power);
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location2 + (i << n_power)] = temp_[i];
        }
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            out1[location2 + ((i + ibase_size) << n_power)] = temp2[i];
        }
    }

    __global__ void cross_multiplication(Data64* in1, Data64* in2, Data64* out,
                                         Modulus64* modulus, int n_power,
                                         int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // decomp size + bsk size

        int location = idx + (idy << n_power);

        Data64 ct0_0 = in1[location];
        Data64 ct0_1 = in1[location + (decomp_size << n_power)];

        Data64 ct1_0 = in2[location];
        Data64 ct1_1 = in2[location + (decomp_size << n_power)];

        Data64 out_0 = OPERATOR_GPU_64::mult(ct0_0, ct1_0, modulus[idy]);
        Data64 out_1_0 = OPERATOR_GPU_64::mult(ct0_0, ct1_1, modulus[idy]);
        Data64 out_1_1 = OPERATOR_GPU_64::mult(ct0_1, ct1_0, modulus[idy]);
        Data64 out_2 = OPERATOR_GPU_64::mult(ct0_1, ct1_1, modulus[idy]);
        Data64 out_1 = OPERATOR_GPU_64::add(out_1_0, out_1_1, modulus[idy]);

        out[location] = out_0;
        out[location + (decomp_size << n_power)] = out_1;
        out[location + (decomp_size << (n_power + 1))] = out_2;
    }

    __global__ void fast_floor(
        Data64* in_baseq_Bsk, Data64* out1, Modulus64* ibase, Modulus64* obase,
        Modulus64 plain_modulus, Data64* inv_punctured_prod_mod_base_array,
        Data64* base_change_matrix_Bsk, Data64* inv_prod_q_mod_Bsk,
        Data64* inv_punctured_prod_mod_B_array, Data64* base_change_matrix_q,
        Data64* base_change_matrix_msk, Data64 inv_prod_B_mod_m_sk,
        Data64* prod_B_mod_q, int n_power, int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // 3

        int location_q =
            idx + ((idy * (ibase_size + obase_size))
                   << n_power); // ibase_size = decomp_modulus_count
        int location_Bsk =
            idx + ((idy * (ibase_size + obase_size)) << n_power) +
            (ibase_size << n_power); // ibase_size = decomp_modulus_count

        Data64 reg_q[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            reg_q[i] =
                OPERATOR_GPU_64::mult(in_baseq_Bsk[location_q + (i << n_power)],
                                      plain_modulus.value, ibase[i]);
            reg_q[i] = OPERATOR_GPU_64::mult(
                reg_q[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        Data64 reg_Bsk[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            reg_Bsk[i] = OPERATOR_GPU_64::mult(
                in_baseq_Bsk[location_Bsk + (i << n_power)],
                plain_modulus.value, obase[i]);
        }

        // for Bsk
        Data64 temp[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp[i] = 0;
            for (int j = 0; j < ibase_size; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    reg_q[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp[i] = OPERATOR_GPU_64::add(temp[i], mult, obase[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data64 temp2 =
                OPERATOR_GPU_64::sub(obase[i].value, temp[i], obase[i]);
            temp2 = OPERATOR_GPU_64::add(temp2, reg_Bsk[i], obase[i]);
            reg_Bsk[i] =
                OPERATOR_GPU_64::mult(temp2, inv_prod_q_mod_Bsk[i], obase[i]);
        }

        Data64 temp3[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size - 1; i++)
        { // only B bases
            temp3[i] = OPERATOR_GPU_64::mult(
                reg_Bsk[i], inv_punctured_prod_mod_B_array[i], obase[i]);
        }

        Data64 temp4[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp4[i] = 0;
#pragma unroll
            for (int j = 0; j < obase_size - 1; j++)
            {
                Data64 temp3_ =
                    OPERATOR_GPU_64::reduce_forced(temp3[j], ibase[i]); // extra

                Data64 mult = OPERATOR_GPU_64::mult(
                    temp3_, base_change_matrix_q[j + (i * (obase_size - 1))],
                    ibase[i]); // extra
                mult = OPERATOR_GPU_64::reduce_forced(mult, ibase[i]); // extra
                temp4[i] = OPERATOR_GPU_64::add(temp4[i], mult, ibase[i]);
            }
        }

        // for m_sk
        temp4[ibase_size] = 0;
#pragma unroll
        for (int j = 0; j < obase_size - 1; j++)
        {
            Data64 mult = OPERATOR_GPU_64::mult(
                temp3[j], base_change_matrix_msk[j], obase[obase_size - 1]);
            temp4[ibase_size] = OPERATOR_GPU_64::add(temp4[ibase_size], mult,
                                                     obase[obase_size - 1]);
        }

        Data64 alpha_sk = OPERATOR_GPU_64::sub(obase[obase_size - 1].value,
                                               reg_Bsk[obase_size - 1],
                                               obase[obase_size - 1]);
        alpha_sk = OPERATOR_GPU_64::add(alpha_sk, temp4[ibase_size],
                                        obase[obase_size - 1]);
        alpha_sk = OPERATOR_GPU_64::mult(alpha_sk, inv_prod_B_mod_m_sk,
                                         obase[obase_size - 1]);

        Data64 m_sk_div_2 = obase[obase_size - 1].value >> 1;

#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            Data64 obase_ = OPERATOR_GPU_64::reduce_forced(
                obase[obase_size - 1].value, ibase[i]);
            Data64 temp4_ = OPERATOR_GPU_64::reduce_forced(temp4[i], ibase[i]);
            Data64 alpha_sk_ =
                OPERATOR_GPU_64::reduce_forced(alpha_sk, ibase[i]);
            if (alpha_sk > m_sk_div_2)
            {
                Data64 inner =
                    OPERATOR_GPU_64::sub(obase_, alpha_sk_, ibase[i]); // extra
                inner = OPERATOR_GPU_64::mult(inner, prod_B_mod_q[i], ibase[i]);
                temp4[i] = OPERATOR_GPU_64::add(temp4_, inner, ibase[i]);
            }
            else
            {
                Data64 inner = OPERATOR_GPU_64::sub(ibase[i].value,
                                                    prod_B_mod_q[i], ibase[i]);
                inner =
                    OPERATOR_GPU_64::mult(inner, alpha_sk_, ibase[i]); // extra
                temp4[i] = OPERATOR_GPU_64::add(temp4_, inner, ibase[i]);
            }
        }

        int location_out =
            idx + ((idy * ibase_size)
                   << n_power); // ibase_size = decomp_modulus_count
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location_out + (i << n_power)] = temp4[i];
        }
    }

    __global__ void threshold_kernel(Data64* plain_in, Data64* output,
                                     Modulus64* modulus,
                                     Data64* plain_upper_half_increment,
                                     Data64 plain_upper_half_threshold,
                                     int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod

        Data64 plain_reg = plain_in[idx];

        if (plain_reg >= plain_upper_half_threshold)
        {
            output[idx + (block_y << n_power)] = OPERATOR_GPU_64::add(
                plain_reg, plain_upper_half_increment[block_y],
                modulus[block_y]); // plain_reg +
                                   // plain_upper_half_increment[block_y];
        }
        else
        {
            output[idx + (block_y << n_power)] = plain_reg;
        }
    }

    __global__ void cipherplain_kernel(Data64* cipher, Data64* plain_in,
                                       Data64* output, Modulus64* modulus,
                                       int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod
        int block_z = blockIdx.z; // cipher size

        int index1 = idx + (block_y << n_power);
        int index2 = index1 + ((decomp_size << n_power) * block_z);

        output[index2] = OPERATOR_GPU_64::mult(cipher[index2], plain_in[index1],
                                               modulus[block_y]);
    }

    __global__ void cipherplain_multiplication_kernel(Data64* in1, Data64* in2,
                                                      Data64* out,
                                                      Modulus64* modulus,
                                                      int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_pt = idx + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 pt = in2[location_pt];

        ct = OPERATOR_GPU_64::mult(ct, pt, modulus[block_y]);
        out[location_ct] = ct;
    }

    __global__ void
    cipher_constant_plain_multiplication_kernel(Data64* in1, double in2,
                                                Data64* out, Modulus64* modulus,
                                                double two_pow_64, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);

        double message_r = in2;

        double coeff_double = round(message_r);
        bool is_negative = signbit(coeff_double);
        coeff_double = fabs(coeff_double);

        // Change Type
        Data64 coeff[2] = {
            static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
            static_cast<std::uint64_t>(coeff_double / two_pow_64)};

        Data64 pt;
        if (is_negative)
        {
            pt = OPERATOR_GPU_64::reduce(coeff, modulus[block_y]);
            pt = OPERATOR_GPU_64::sub(modulus[block_y].value, pt,
                                      modulus[block_y]);
        }
        else
        {
            pt = OPERATOR_GPU_64::reduce(coeff, modulus[block_y]);
        }

        Data64 ct = in1[location_ct];

        ct = OPERATOR_GPU_64::mult(ct, pt, modulus[block_y]);
        out[location_ct] = ct;
    }

    // Multiply-accumulate over the baby-step ciphertexts at positions
    // given by ct_indices[i].
    // ct_indices[i] is the baby-step index; the ciphertext offset is
    //   ct_indices[i] * ct_stride  (ct_stride = 2 * current_decomp_count * N).
    // Grid: dim3((n >> 8), pql_count, 2), 256
    __global__ void cipherplain_multiply_accumulate_indexed_kernel(
        Data64* baby_results, Data64* plaintexts, Data64* out,
        Modulus64* modulus, int* ct_indices, int iteration_count,
        int current_decomp_count, int first_decomp_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int block_y = blockIdx.y;
        int block_z = blockIdx.z;

        int location_out =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_pt = idx + (block_y << n_power);

        int ct_stride = (current_decomp_count << (n_power + 1));
        int offset_pt = (first_decomp_count << n_power);

        Data64 sum_ctpt = 0ULL;
        for (int i = 0; i < iteration_count; i++)
        {
            int ct_offset = ct_indices[i] * ct_stride;
            Data64 ct = baby_results[location_out + ct_offset];
            Data64 pt = plaintexts[location_pt + (i * offset_pt)];
            Data64 mul_ctpt = OPERATOR_GPU_64::mult(ct, pt, modulus[block_y]);
            sum_ctpt =
                OPERATOR_GPU_64::add(sum_ctpt, mul_ctpt, modulus[block_y]);
        }

        out[location_out] = sum_ctpt;
    }

    __global__ void cipher_div_by_i_kernel(Data64* in1, Data64* out,
                                           Data64* ntt_table,
                                           Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_psi = 1 + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 psi = ntt_table[location_psi];

        if (idx < (1 << (n_power - 1)))
        {
            Data64 zero = 0;
            Data64 neg_psi = OPERATOR_GPU_64::sub(zero, psi, modulus[block_y]);
            ct = OPERATOR_GPU_64::mult(ct, neg_psi, modulus[block_y]);
        }
        else
        {
            ct = OPERATOR_GPU_64::mult(ct, psi, modulus[block_y]);
        }
        out[location_ct] = ct;
    }

    __global__ void cipher_mult_by_i_kernel(Data64* in1, Data64* out,
                                            Data64* ntt_table,
                                            Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_psi = 1 + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 psi = ntt_table[location_psi];

        if (idx < (1 << (n_power - 1)))
        {
            ct = OPERATOR_GPU_64::mult(ct, psi, modulus[block_y]);
        }
        else
        {
            Data64 zero = 0;
            Data64 neg_psi = OPERATOR_GPU_64::sub(zero, psi, modulus[block_y]);
            ct = OPERATOR_GPU_64::mult(ct, neg_psi, modulus[block_y]);
        }
        out[location_ct] = ct;
    }

    __global__ void cipher_mult_by_gaussian_integer_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* out,
        Data64* ntt_table, Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_psi = 1 + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 psi = ntt_table[location_psi];

        Data64 c_real = real_rns[block_y];
        Data64 c_imag = imag_rns[block_y];

        Data64 const_imag =
            OPERATOR_GPU_64::mult(c_imag, psi, modulus[block_y]);

        if (idx < (1 << (n_power - 1)))
        {
            Data64 scaled_const =
                OPERATOR_GPU_64::add(c_real, const_imag, modulus[block_y]);
            ct = OPERATOR_GPU_64::mult(ct, scaled_const, modulus[block_y]);
        }
        else
        {
            Data64 scaled_const =
                OPERATOR_GPU_64::sub(c_real, const_imag, modulus[block_y]);
            ct = OPERATOR_GPU_64::mult(ct, scaled_const, modulus[block_y]);
        }
        out[location_ct] = ct;
    }

    __global__ void cipher_add_by_gaussian_integer_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* out,
        Data64* ntt_table, Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_psi = 1 + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 psi = ntt_table[location_psi];

        Data64 c_real = real_rns[block_y];
        Data64 c_imag = imag_rns[block_y];

        Data64 const_imag =
            OPERATOR_GPU_64::mult(c_imag, psi, modulus[block_y]);

        if (block_z == 0)
        {
            if (idx < (1 << (n_power - 1)))
            {
                Data64 scaled_const =
                    OPERATOR_GPU_64::add(c_real, const_imag, modulus[block_y]);
                ct = OPERATOR_GPU_64::add(ct, scaled_const, modulus[block_y]);
            }
            else
            {
                Data64 scaled_const =
                    OPERATOR_GPU_64::sub(c_real, const_imag, modulus[block_y]);
                ct = OPERATOR_GPU_64::add(ct, scaled_const, modulus[block_y]);
            }
        }
        out[location_ct] = ct;
    }

    __global__ void cipher_mult_by_gaussian_integer_and_add_kernel(
        Data64* in1, Data64* real_rns, Data64* imag_rns, Data64* accumulator,
        Data64* ntt_table, Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count (for ct0, ct1, etc.)

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_psi = 1 + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 acc = accumulator[location_ct];
        Data64 psi = ntt_table[location_psi];

        Data64 c_real = real_rns[block_y];
        Data64 c_imag = imag_rns[block_y];

        Data64 const_imag =
            OPERATOR_GPU_64::mult(c_imag, psi, modulus[block_y]);

        Data64 mult_result;
        if (idx < (1 << (n_power - 1)))
        {
            Data64 scaled_const =
                OPERATOR_GPU_64::add(c_real, const_imag, modulus[block_y]);
            mult_result =
                OPERATOR_GPU_64::mult(ct, scaled_const, modulus[block_y]);
        }
        else
        {
            Data64 scaled_const =
                OPERATOR_GPU_64::sub(c_real, const_imag, modulus[block_y]);
            mult_result =
                OPERATOR_GPU_64::mult(ct, scaled_const, modulus[block_y]);
        }

        accumulator[location_ct] =
            OPERATOR_GPU_64::add(acc, mult_result, modulus[block_y]);
    }

} // namespace heongpu