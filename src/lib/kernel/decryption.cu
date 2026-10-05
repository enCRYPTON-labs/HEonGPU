// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/decryption.cuh>

namespace heongpu
{
    __global__ void sk_multiplication(Data64* ct1, Data64* sk, Data64* output,
                                      Modulus64* modulus, int n_power,
                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_1 = ct1[index];
        Data64 sk_ = sk[index];

        output[index] = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);
    }

    __global__ void sk_multiplicationx3(Data64* ct1, Data64* sk,
                                        Modulus64* modulus, int n_power,
                                        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_1 = ct1[index];
        Data64 sk_ = sk[index];
        ct1[index] = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);

        Data64 ct_2 = ct1[index + (decomp_mod_count << n_power)];
        Data64 sk2_ = OPERATOR_GPU_64::mult(sk_, sk_, modulus[block_y]);
        ct1[index + (decomp_mod_count << n_power)] =
            OPERATOR_GPU_64::mult(ct_2, sk2_, modulus[block_y]);
    }

    __global__ void decryption_kernel(Data64* ct0, Data64* ct1, Data64* plain,
                                      Modulus64* modulus, Modulus64 plain_mod,
                                      Modulus64 gamma, Data64* Qi_t,
                                      Data64* Qi_gamma, Data64* Qi_inverse,
                                      Data64 mulq_inv_t, Data64 mulq_inv_gamma,
                                      Data64 inv_gamma, int n_power,
                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt =
                OPERATOR_GPU_64::add(ct0[location], ct1[location], modulus[i]);

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void decryption_kernelx3(Data64* ct0, Data64* ct1, Data64* ct2,
                                        Data64* plain, Modulus64* modulus,
                                        Modulus64 plain_mod, Modulus64 gamma,
                                        Data64* Qi_t, Data64* Qi_gamma,
                                        Data64* Qi_inverse, Data64 mulq_inv_t,
                                        Data64 mulq_inv_gamma, Data64 inv_gamma,
                                        int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt =
                OPERATOR_GPU_64::add(ct0[location], ct1[location], modulus[i]);

            mt = OPERATOR_GPU_64::add(mt, ct2[location], modulus[i]);

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void coeff_multadd(Data64* input1, Data64* input2,
                                  Data64* output, Modulus64 plain_mod,
                                  Modulus64* modulus, int n_power,
                                  int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_0 = input1[index];
        Data64 ct_1 = input2[index];

        ct_0 = OPERATOR_GPU_64::add(ct_1, ct_0, modulus[block_y]);
        ct_0 = OPERATOR_GPU_64::mult(ct_0, plain_mod.value, modulus[block_y]);

        output[index] = ct_0;
    }

    __global__ void compose_kernel(Data64* input, Data64* output,
                                   Modulus64* modulus, Data64* Mi_inv,
                                   Data64* Mi, Data64* decryption_modulus,
                                   int coeff_modulus_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 compose_result[50]; // TODO: Define size as global variable
        Data64 big_integer_result[50]; // TODO: Define size as global variable

        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data64 base = input[idx + (i << n_power)];
            Data64 temp = OPERATOR_GPU_64::mult(base, Mi_inv[i], modulus[i]);

            biginteger::multiply(Mi + (i * coeff_modulus_count),
                                 coeff_modulus_count, temp, big_integer_result,
                                 coeff_modulus_count);

            int carry = biginteger::add_inplace(
                compose_result, big_integer_result, coeff_modulus_count);

            bool check = biginteger::is_greater_or_equal(
                compose_result, decryption_modulus, coeff_modulus_count);

            if (check)
            {
                biginteger::sub2(compose_result, decryption_modulus,
                                 coeff_modulus_count, compose_result);
            }
        }

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            output[idx + (i << n_power)] = compose_result[i];
        }
    }

    __global__ void find_max_norm_kernel(Data64* input, Data64* output,
                                         Data64* upper_half_threshold,
                                         Data64* decryption_modulus,
                                         int coeff_modulus_count, int n_power)
    {
        int idx = threadIdx.x;
        int offset = blockDim.x;
        int iteration_count = (1 << n_power) / offset;

        Data64 big_integer_input[50]; // TODO: Define size as global variable
        Data64 big_integer_temp[50]; // TODO: Define size as global variable

        // Data64 *big_integer_input = (Data64 *)alloca(20 * sizeof(Data64));
        // Data64 *big_integer_temp = (Data64 *)alloca(20 * sizeof(Data64));

        biginteger::set_zero(big_integer_temp, coeff_modulus_count);

        for (int i = 0; i < iteration_count; i++)
        {
            for (int j = 0; j < coeff_modulus_count; j++)
            {
                big_integer_input[j] =
                    input[idx + (i * offset) + (j << n_power)];
            }

            bool check = biginteger::is_greater_or_equal(
                big_integer_input, upper_half_threshold, coeff_modulus_count);

            if (check)
            {
                biginteger::sub2(decryption_modulus, big_integer_input,
                                 coeff_modulus_count, big_integer_input);
            }

            check = biginteger::is_greater(big_integer_input, big_integer_temp,
                                           coeff_modulus_count);

            if (check)
            {
                biginteger::set(big_integer_input, coeff_modulus_count,
                                big_integer_temp);
            }

            __syncthreads();
        }

        extern __shared__ Data64 shared_memory[];

        __syncthreads();

        for (int offset_in = blockDim.x >> 1; offset_in > 0; offset_in >>= 1)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                if (idx >= offset_in && idx < (offset_in << 1))
                {
                    shared_memory[idx - offset_in] = big_integer_temp[i];
                }

                __syncthreads();

                if (idx < offset_in)
                {
                    big_integer_input[i] = shared_memory[idx];
                }

                __syncthreads();
            }

            if (idx < offset_in &&
                biginteger::is_greater(big_integer_input, big_integer_temp,
                                      coeff_modulus_count))
            {
                biginteger::set(big_integer_input, coeff_modulus_count,
                                big_integer_temp);
            }

            __syncthreads();
        }

        if (idx == 0)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                output[i] = big_integer_temp[i];
            }
        }
    }

    __global__ void sk_multiplication_ckks(Data64* ciphertext,
                                           Data64* plaintext, Data64* sk,
                                           Modulus64* modulus, int n_power,
                                           int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_0 = ciphertext[index];
        Data64 ct_1 = ciphertext[index + (decomp_mod_count << n_power)];
        Data64 sk_ = sk[index];

        ct_1 = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);
        ct_0 = OPERATOR_GPU_64::add(ct_1, ct_0, modulus[block_y]);

        plaintext[index] = ct_0;
    }

    //////////////////
    //////////////////

    __global__ void decryption_fusion_bfv_kernel(
        Data64* ct, Data64* plain, Modulus64* modulus, Modulus64 plain_mod,
        Modulus64 gamma, Data64* Qi_t, Data64* Qi_gamma, Data64* Qi_inverse,
        Data64 mulq_inv_t, Data64 mulq_inv_gamma, Data64 inv_gamma, int n_power,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt = ct[location];

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void decrypt_lwe_kernel(int32_t* sk, int32_t* input_a,
                                       int32_t* input_b, int32_t* output, int n,
                                       int k)
    {
        extern __shared__ uint32_t sdata[];
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int lane = idx & (warpSize - 1);
        int wid = idx >> 5;
        int n_warps = (blockDim.x + warpSize - 1) >> 5;

        int base = block_x * n;
        uint32_t local_sum = 0;
        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t secret_key = sk[i];
            uint32_t r = input_a[base + i];
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
                output[block_x] =
                    static_cast<int32_t>(input_b[block_x] - block_sum);
            }
        }
    }

    __global__ void col_boot_dec_mul_with_sk(const Data64* ct1, const Data64* a,
                                             const Data64* sk, Data64* output,
                                             const Modulus64* modulus,
                                             int n_power, int decomp_mod_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        const int block_y = blockIdx.y; // decomp_mod_count
        const int block_z = blockIdx.z; // 2

        int in_index = idx + (block_y << n_power);
        int out_index = in_index + ((decomp_mod_count * block_z) << n_power);

        const Modulus64 mod = modulus[block_y];

        const Data64 sk_ = sk[in_index];
        Data64 result;

        if (block_z == 0)
        {
            // c1 * sk mod q
            Data64 ct_1_ = ct1[in_index];
            result = OPERATOR_GPU_64::mult(ct_1_, sk_, mod);
        }
        else
        {
            // −(a * sk) mod q
            Data64 zero = 0ULL;
            Data64 a_ = a[in_index];
            result = OPERATOR_GPU_64::mult(a_, sk_, mod);
            result = OPERATOR_GPU_64::sub(zero, result, mod);
        }

        output[out_index] = result;
    }

    __global__ void col_boot_add_random_and_errors(
        Data64* ct, const Data64* errors, const Data64* random_plain,
        const Modulus64* modulus, Modulus64 plain_mod, Data64 Q_mod_t,
        Data64 upper_threshold, Data64* coeffdiv_plain, int n_power,
        int decomp_mod_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        const int block_y = blockIdx.y; // decomp_mod_count
        const int block_z = blockIdx.z; // 2

        int in_index = idx + (block_y << n_power) +
                       ((decomp_mod_count * block_z) << n_power);

        const Modulus64 mod = modulus[block_y];
        Data64 ct_ = ct[in_index];
        Data64 error = errors[in_index];

        // Compute ∆M
        Data64 random_message = random_plain[idx];
        Data64 fix = random_message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);
        Data64 delta_m =
            OPERATOR_GPU_64::mult(random_message, coeffdiv_plain[block_y], mod);
        delta_m = OPERATOR_GPU_64::add(delta_m, fix, mod);

        // Add error term
        ct_ = OPERATOR_GPU_64::add(ct_, error, mod);

        if (block_z == 0)
        {
            ct_ = OPERATOR_GPU_64::sub(ct_, delta_m, mod);
        }
        else
        {
            ct_ = OPERATOR_GPU_64::add(ct_, delta_m, mod);
        }

        ct[in_index] = ct_;
    }

    __global__ void col_boot_enc(Data64* ct, const Data64* h,
                                 const Data64* random_plain,
                                 const Modulus64* modulus, Modulus64 plain_mod,
                                 Data64 Q_mod_t, Data64 upper_threshold,
                                 Data64* coeffdiv_plain, int n_power,
                                 int decomp_mod_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        const int block_y = blockIdx.y; // decomp_mod_count

        int in_index = idx + (block_y << n_power);

        const Modulus64 mod = modulus[block_y];
        Data64 h_ = h[in_index];

        // Compute ∆M
        Data64 random_message = random_plain[idx];
        Data64 fix = random_message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);
        Data64 delta_m =
            OPERATOR_GPU_64::mult(random_message, coeffdiv_plain[block_y], mod);
        delta_m = OPERATOR_GPU_64::add(delta_m, fix, mod);

        Data64 ct_ = OPERATOR_GPU_64::add(h_, delta_m, mod);

        ct[in_index] = ct_;
    }

    __global__ void col_boot_dec_mul_with_sk_ckks(
        const Data64* ct1, const Data64* a, const Data64* sk, Data64* output,
        const Modulus64* modulus, int n_power, int decomp_mod_count,
        int current_decomp_mod_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        const int block_y =
            blockIdx.y; // current_decomp_mod_count + decomp_mod_count

        int in_index = idx + (block_y << n_power);
        Data64 result = 0ULL;

        if (block_y < current_decomp_mod_count)
        {
            const Modulus64 mod = modulus[block_y];
            const Data64 sk_ = sk[in_index];

            // c1 * sk mod q
            Data64 ct_1_ = ct1[in_index];
            result = OPERATOR_GPU_64::mult(ct_1_, sk_, mod);
        }
        else
        {
            int offset_block = block_y - current_decomp_mod_count;
            const Modulus64 mod = modulus[offset_block];
            int m_in_index = idx + (offset_block << n_power);
            const Data64 sk_ = sk[m_in_index];

            // −(a * sk) mod q
            Data64 zero = 0ULL;
            Data64 a_ = a[m_in_index];
            result = OPERATOR_GPU_64::mult(a_, sk_, mod);
            result = OPERATOR_GPU_64::sub(zero, result, mod);
        }

        output[in_index] = result;
    }

    __global__ void col_boot_add_random_and_errors_ckks(
        Data64* ct, const Data64* error0, const Data64* error1,
        const Data64* random_plain, const Modulus64* modulus, int n_power,
        int decomp_mod_count, int current_decomp_mod_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        const int block_y =
            blockIdx.y; // current_decomp_mod_count + decomp_mod_count

        Data64 ct_ = 0ULL;
        int in_index = idx + (block_y << n_power);

        if (block_y < current_decomp_mod_count)
        {
            const Modulus64 mod = modulus[block_y];

            ct_ = ct[in_index];
            Data64 error = error0[in_index];
            Data64 random_message = random_plain[in_index];

            // Add error term
            ct_ = OPERATOR_GPU_64::add(ct_, error, mod);

            // Add random message
            ct_ = OPERATOR_GPU_64::sub(ct_, random_message, mod);
        }
        else
        {
            int offset_block = block_y - current_decomp_mod_count;
            const Modulus64 mod = modulus[offset_block];
            int m_in_index = idx + (offset_block << n_power);

            ct_ = ct[in_index];
            Data64 error = error1[m_in_index];
            Data64 random_message = random_plain[m_in_index];

            // Add error term
            ct_ = OPERATOR_GPU_64::add(ct_, error, mod);

            // Add random message
            ct_ = OPERATOR_GPU_64::add(ct_, random_message, mod);
        }

        ct[in_index] = ct_;
    }

} // namespace heongpu