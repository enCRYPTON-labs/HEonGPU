// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/encryption.cuh>

namespace heongpu
{
    __global__ void pk_u_kernel(Data64* pk, Data64* u, Data64* pk_u,
                                Modulus64* modulus, int n_power,
                                int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count
        int block_z = blockIdx.z; // 2

        Data64 pk_ = pk[idx + (block_y << n_power) +
                        ((rns_mod_count << n_power) * block_z)];
        Data64 u_ = u[idx + (block_y << n_power)];

        Data64 pk_u_ = OPERATOR_GPU_64::mult(pk_, u_, modulus[block_y]);

        pk_u[idx + (block_y << n_power) +
             ((rns_mod_count << n_power) * block_z)] = pk_u_;
    }

    __global__ void enc_div_lastq_kernel(
        Data64* pk, Data64* e, Data64* plain, Data64* ct, Modulus64* modulus,
        Data64 half, Data64* half_mod, Data64* last_q_modinv,
        Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
        Data64* coeffdiv_plain, int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_pk = pk[idx + (decomp_mod_count << n_power) +
                            (((decomp_mod_count + 1) << n_power) * block_z)];
        Data64 last_e = e[idx + (decomp_mod_count << n_power) +
                          (((decomp_mod_count + 1) << n_power) * block_z)];

        last_pk =
            OPERATOR_GPU_64::add(last_pk, last_e, modulus[decomp_mod_count]);

        last_pk =
            OPERATOR_GPU_64::add(last_pk, half, modulus[decomp_mod_count]);

        last_pk = OPERATOR_GPU_64::reduce_forced(last_pk, modulus[block_y]);

        last_pk =
            OPERATOR_GPU_64::sub(last_pk, half_mod[block_y], modulus[block_y]);

        Data64 input_ = pk[idx + (block_y << n_power) +
                           (((decomp_mod_count + 1) << n_power) * block_z)];

        //
        Data64 e_ = e[idx + (block_y << n_power) +
                      (((decomp_mod_count + 1) << n_power) * block_z)];
        input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);
        //

        input_ = OPERATOR_GPU_64::sub(input_, last_pk, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        if (block_z == 0)
        {
            Data64 message = plain[idx];
            Data64 fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data64 ct_0 = OPERATOR_GPU_64::mult(
                message, coeffdiv_plain[block_y], modulus[block_y]);
            ct_0 = OPERATOR_GPU_64::add(ct_0, fix, modulus[block_y]);

            input_ = OPERATOR_GPU_64::add(input_, ct_0, modulus[block_y]);

            ct[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = input_;
        }
        else
        {
            ct[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = input_;
        }
    }

    __global__ void
    enc_div_lastq_bfv_kernel(Data64* pk, Data64* e, Data64* plain, Data64* ct,
                             Modulus64* modulus, Data64* half, Data64* half_mod,
                             Data64* last_q_modinv, Modulus64 plain_mod,
                             Data64 Q_mod_t, Data64 upper_threshold,
                             Data64* coeffdiv_plain, int n_power,
                             int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_pk[15];
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_pk_ = pk[idx + ((Q_size + i) << n_power) +
                                 ((Q_prime_size << n_power) * block_z)];
            Data64 last_e_ = e[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
            last_pk[i] =
                OPERATOR_GPU_64::add(last_pk_, last_e_, modulus[Q_size + i]);
        }

        Data64 input_ = pk[idx + (block_y << n_power) +
                           ((Q_prime_size << n_power) * block_z)];
        Data64 e_ = e[idx + (block_y << n_power) +
                      ((Q_prime_size << n_power) * block_z)];
        input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_pk_add_half_ = last_pk[(P_size - 1 - i)];
            last_pk_add_half_ = OPERATOR_GPU_64::add(
                last_pk_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::reduce_forced(
                    last_pk_add_half_, modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_pk[j], temp1,
                                             modulus[Q_size + j]);

                last_pk[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::reduce_forced(last_pk_add_half_,
                                                          modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data64 message = plain[idx];
            Data64 fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data64 ct_0 = OPERATOR_GPU_64::mult(
                message, coeffdiv_plain[block_y], modulus[block_y]);
            ct_0 = OPERATOR_GPU_64::add(ct_0, fix, modulus[block_y]);

            input_ = OPERATOR_GPU_64::add(input_, ct_0, modulus[block_y]);

            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
                input_;
        }
        else
        {
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
                input_;
        }
    }

    __global__ void enc_div_lastq_ckks_kernel(Data64* pk, Data64* e, Data64* ct,
                                              Modulus64* modulus, Data64* half,
                                              Data64* half_mod,
                                              Data64* last_q_modinv,
                                              int n_power, int Q_prime_size,
                                              int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_pk[15];
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_pk_ = pk[idx + ((Q_size + i) << n_power) +
                                 ((Q_prime_size << n_power) * block_z)];
            Data64 last_e_ = e[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
            last_pk[i] =
                OPERATOR_GPU_64::add(last_pk_, last_e_, modulus[Q_size + i]);
        }

        Data64 input_ = pk[idx + (block_y << n_power) +
                           ((Q_prime_size << n_power) * block_z)];
        Data64 e_ = e[idx + (block_y << n_power) +
                      ((Q_prime_size << n_power) * block_z)];
        input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_pk_add_half_ = last_pk[(P_size - 1 - i)];
            last_pk_add_half_ = OPERATOR_GPU_64::add(
                last_pk_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);

            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::reduce(last_pk_add_half_,
                                                       modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_pk[j], temp1,
                                             modulus[Q_size + j]);

                last_pk[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            // Data64 temp1 = OPERATOR_GPU_64::reduce(last_pk_add_half_,
            // modulus[block_y]);
            Data64 temp1 = OPERATOR_GPU_64::reduce_forced(last_pk_add_half_,
                                                          modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            input_;
    }

    __global__ void cipher_message_add_kernel(Data64* ciphertext,
                                              Data64* plaintext,
                                              Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)

        Data64 ct_0 = ciphertext[idx + (block_y << n_power)];
        Data64 plaintext_ = plaintext[idx + (block_y << n_power)];

        ct_0 = OPERATOR_GPU_64::add(ct_0, plaintext_, modulus[block_y]);

        ciphertext[idx + (block_y << n_power)] = ct_0;
    }

    __global__ void encrypt_lwe_kernel(const int32_t* sk,
                                       const int32_t* output_a,
                                       int32_t* output_b, int n, int k)
    {
        extern __shared__ uint32_t sdata[];
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int lane = idx & (warpSize - 1);
        int wid = idx >> 5;
        int n_warps = (blockDim.x + warpSize - 1) >> 5;

        for (int seg = block_x; seg < k; seg += gridDim.x)
        {
            int base = seg * n;
            uint32_t local_sum = 0;
            for (int i = idx; i < n; i += blockDim.x)
            {
                uint32_t secret_key = sk[i];
                uint32_t r = output_a[base + i];
                local_sum += (uint32_t) (r * secret_key);
            }

            uint32_t warp_sum = warp_reduce(local_sum);

            if (lane == 0)
                sdata[wid] = warp_sum;
            __syncthreads();

            if (wid == 0)
            {
                uint32_t block_sum = (lane < n_warps ? sdata[lane] : 0);
                block_sum = warp_reduce(block_sum);
                if (lane == 0)
                {
                    output_b[seg] =
                        static_cast<int32_t>(block_sum + output_b[seg]);
                }
            }
            __syncthreads();
        }
    }

} // namespace heongpu
