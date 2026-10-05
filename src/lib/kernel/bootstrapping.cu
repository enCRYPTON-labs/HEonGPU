// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/bootstrapping.cuh>

namespace heongpu
{

    __device__ int exponent_calculation(int& index, int& n)
    {
        Data64 result = 1ULL;
        Data64 five = 5ULL;
        Data64 mod = (n << 2) - 1;

        int bits = 32 - __clz(index);
        for (int i = bits - 1; i > -1; i--)
        {
            result = (result * result) & mod;

            if (((index >> i) & 1u))
            {
                result = (result * five) & mod;
            }
        }

        return result;
    }

    __device__ int matrix_location(int& index)
    {
        if (index == 0)
        {
            return 0;
        }

        return (3 * index) - 1;
    }

    __device__ int matrix_reverse_location(int& index)
    {
        int total = (gridDim.y - 1) * 3;
        if (index == 0)
        {
            return total;
        }

        return total - (3 * index);
    }

    __global__ void E_diagonal_generate_kernel(Complex64* output, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int block_y = blockIdx.y; // matrix index
        int logk = block_y + 1;
        int output_location = matrix_location(block_y);

        int n = 1 << n_power;
        int v_size = 1 << (n_power - logk);

        int index1 = idx & ((v_size << 1) - 1);
        int index2 = index1 >> (n_power - logk);
        Complex64 W1(1.0, 0.0);
        Complex64 W2(0.0, 0.0);
        Complex64 W3(0.0, 0.0);

        if (block_y == 0)
        {
            double angle = M_PI / (v_size << 2);
            Complex64 omega_4n(cos(angle), sin(angle));
            int expo = exponent_calculation(index1, n);

            Complex64 W = omega_4n.exp(expo);
            Complex64 W_neg = W; // W.negate();

            if (index2 == 1)
            {
                W1 = W_neg;
                W2 = Complex64(1.0, 0.0);
            }
            else
            {
                W2 = W;
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
        }
        else
        {
            double angle = M_PI / (v_size << 2);
            Complex64 omega_4n(cos(angle), sin(angle));
            int expo = exponent_calculation(index1, n);

            Complex64 W = omega_4n.exp(expo);
            Complex64 W_neg = W; // W.negate();

            if (index2 == 1)
            {
                W1 = W_neg;
                W3 = Complex64(1.0, 0.0);
            }
            else
            {
                W2 = W;
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
            output[((output_location + 2) << n_power) + idx] = W3;
        }
    }

    __global__ void E_diagonal_inverse_generate_kernel(Complex64* output,
                                                       int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int block_y = blockIdx.y; // matrix index
        int logk = block_y + 1;
        int output_location = matrix_reverse_location(block_y);

        int n = 1 << n_power;
        int v_size = 1 << (n_power - logk);

        int index1 = idx & ((v_size << 1) - 1);
        int index2 = index1 >> (n_power - logk);
        Complex64 W1(0.5, 0.0);
        Complex64 W2(0.5, 0.0);
        Complex64 W3(0.0, 0.0);

        if (block_y == 0)
        {
            if (index2 == 1)
            {
                double angle = M_PI / (v_size << 2);
                Complex64 omega_4n(cos(angle), sin(angle));
                int expo = exponent_calculation(index1, n);
                W1 = omega_4n.inverse();
                W1 = W1.exp(expo);
                W1 = W1 / Complex64(2.0, 0.0);
                W2 = W1.negate();
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
        }
        else
        {
            if (index2 == 1)
            {
                double angle = M_PI / (v_size << 2);
                Complex64 omega_4n(cos(angle), sin(angle));
                int expo = exponent_calculation(index1, n);
                W1 = omega_4n.inverse();
                W1 = W1.exp(expo);
                W1 = W1 / Complex64(2.0, 0.0);
                W2 = Complex64(0.0, 0.0);
                W3 = W1.negate();
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
            output[((output_location + 2) << n_power) + idx] = W3;
        }
    }

    __global__ void E_diagonal_inverse_matrix_mult_single_kernel(
        Complex64* input, Complex64* output, bool last, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (last)
        {
            for (int i = 0; i < 2; i++)
            {
                output[idx + (i << n_power)] = input[idx + (i << n_power)];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                output[idx + (i << n_power)] = input[idx + (i << n_power)];
            }
        }
    }

    __global__ void E_diagonal_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first1, bool first2, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int offset = first1 ? 2 : 3;
        int L_matrix_loc_ = offset + (3 * mul_index);
        int L_matrix_size = 3;

        int R_matrix_counter_ = R_matrix_counter;
        int output_index_counter_ = output_index_counter;
        int iter_R_m = iteration_count;
        if (first2)
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m = input[idx + (i << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];

                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
        else
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m =
                    temp[idx +
                         (input_index[R_matrix_counter_ - offset] << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];

                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
    }

    __global__ void E_diagonal_inverse_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first, bool last, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int L_matrix_loc_ = 3 + (3 * mul_index);
        int L_matrix_size = (last) ? 2 : 3;

        int R_matrix_counter_ = R_matrix_counter;
        int output_index_counter_ = output_index_counter;
        int iter_R_m = iteration_count;
        if (first)
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m = input[idx + (i << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];
                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
        else
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m =
                    temp[idx + (input_index[R_matrix_counter_ - 3] << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];
                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
    }

    __global__ void complex_vector_scale_kernel(Complex64* data,
                                                Complex64 scaling, int n_power)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x; // index within each vector
        int idy = blockIdx.y; // matrix index

        int location = idx + (idy << n_power);

        data[location] = data[location] * scaling;
    }

    __global__ void vector_rotate_kernel(Complex64* input, Complex64* output,
                                         int rotate_index, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        Complex64 rotated = rotated_access(input, rotate_index, idx, n_power);

        output[idx] = rotated;
    }

    // TODO: implement it for multiple RNS prime (currently it only works for
    // single prime)
    __global__ void mod_raise_kernel(Data64* input, Data64* output,
                                     Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location_input = idx + (idz << n_power);
        int location_output =
            idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        Data64 q0 = modulus[0].value;
        Data64 qi = modulus[idy].value;

        // Get coefficient from level 0
        Data64 coeff = input[location_input];

        // Centered reduction around q0
        // If coeff >= q0/2, use negative representation: coeff = q0 - coeff
        Data64 pos = 1;
        Data64 neg = 0;
        if (coeff >= (q0 >> 1))
        {
            coeff = q0 - coeff;
            pos = 0;
            neg = 1;
        }

        if (idy == 0)
        {
            output[location_output] = input[location_input];
        }
        else
        {
            Data64 tmp = OPERATOR_GPU_64::reduce_forced(coeff, modulus[idy]);
            output[location_output] = tmp * pos + (qi - tmp) * neg;
        }
    }

    __global__ void
    tfhe_nand_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                              int32_t* input1_a, int32_t* input1_b,
                              int32_t* input2_a, int32_t* input2_b,
                              int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a - input1_a_reg;
            local_a = local_a - input2_a_reg;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b - input1_b_reg;
            local_b = local_b - input2_b_reg;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_and_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a + input1_a_reg;
            local_a = local_a + input2_a_reg;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b + input1_b_reg;
            local_b = local_b + input2_b_reg;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_and_first_not_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                                       int32_t* input1_a, int32_t* input1_b,
                                       int32_t* input2_a, int32_t* input2_b,
                                       int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a - input1_a_reg;
            local_a = local_a + input2_a_reg;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b - input1_b_reg;
            local_b = local_b + input2_b_reg;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_nor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a - input1_a_reg;
            local_a = local_a - input2_a_reg;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b - input1_b_reg;
            local_b = local_b - input2_b_reg;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_or_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                            int32_t* input1_a, int32_t* input1_b,
                            int32_t* input2_a, int32_t* input2_b,
                            int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a + input1_a_reg;
            local_a = local_a + input2_a_reg;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b + input1_b_reg;
            local_b = local_b + input2_b_reg;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_xnor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                              int32_t* input1_a, int32_t* input1_b,
                              int32_t* input2_a, int32_t* input2_b,
                              int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a - input1_a_reg;
            local_a = local_a - input2_a_reg;
            local_a = 2 * local_a;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b - (2 * input1_b_reg);
            local_b = local_b - (2 * input2_b_reg);
            local_b = local_b;

            output_b[block_x] = local_b;
        }
    }

    __global__ void
    tfhe_xor_pre_comp_kernel(int32_t* output_a, int32_t* output_b,
                             int32_t* input1_a, int32_t* input1_b,
                             int32_t* input2_a, int32_t* input2_b,
                             int32_t encoded, int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t local_a = 0;
            uint32_t input1_a_reg = input1_a[offset + i];
            uint32_t input2_a_reg = input2_a[offset + i];

            local_a = local_a + input1_a_reg;
            local_a = local_a + input2_a_reg;
            local_a = 2 * local_a;

            output_a[offset + i] = local_a;
        }

        if (idx == 0)
        {
            uint32_t local_b = encoded;
            uint32_t input1_b_reg = input1_b[block_x];
            uint32_t input2_b_reg = input2_b[block_x];

            local_b = local_b + (2 * input1_b_reg);
            local_b = local_b + (2 * input2_b_reg);
            local_b = local_b;

            output_b[block_x] = local_b;
        }
    }

    __global__ void tfhe_not_comp_kernel(int32_t* output_a, int32_t* output_b,
                                         int32_t* input1_a, int32_t* input1_b,
                                         int n)
    {
        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset = block_x * n;

        for (int i = idx; i < n; i += blockDim.x)
        {
            uint32_t input1_a_reg = input1_a[offset + i];

            input1_a_reg = -input1_a_reg;

            output_a[offset + i] = input1_a_reg;
        }

        if (idx == 0)
        {
            uint32_t input1_b_reg = input1_b[block_x];

            input1_b_reg = -input1_b_reg;

            output_b[block_x] = input1_b_reg;
        }
    }

    __device__ int32_t torus_modulus_switch_log(int32_t& input,
                                                int& modulus_log)
    {
        uint64_t range_log = 63 - modulus_log;
        uint64_t half_range = 1ULL << (range_log - 1);
        uint64_t result64 =
            (static_cast<uint64_t>(static_cast<uint32_t>(input)) << 32) +
            half_range;

        int32_t result = static_cast<int32_t>(result64 >> range_log);

        return result;
    }

    __global__ void tfhe_bootstrapping_kernel(
        const int32_t* input_a, const int32_t* input_b, int32_t* output,
        const Data64* boot_key,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus,
        const int32_t encoded, const int32_t bk_offset, const int32_t bk_mask,
        const int32_t bk_half, int n, int N, int N_power, int k, int bk_bit,
        int bk_length)
    {
        __shared__ uint32_t sdata32[1024];
        __shared__ Data64 sdata64[1024];

        int idx = threadIdx.x;
        int block_x = blockIdx.x;

        int offset_lwe = block_x * n;

        const Modulus64 modulus_reg = modulus;
        const Data64 threshold = modulus_reg.value >> 1;
        const Ninverse64 n_inverse_reg = n_inverse;

        int32_t input_b_i = input_b[block_x];
        int32_t input_b_i_N = torus_modulus_switch_log(input_b_i, N_power);
        input_b_i_N = static_cast<int32_t>(N << 1) - input_b_i_N;

        int32_t2 accum[4];
        int32_t2 accum2[4];
        int32_t encoded_reg = encoded;

        if (input_b_i_N < N)
        {
            accum[k].value[0] =
                (idx < input_b_i_N) ? (-encoded_reg) : encoded_reg;
            accum[k].value[1] = ((idx + blockDim.x) < input_b_i_N)
                                    ? (-encoded_reg)
                                    : encoded_reg;
        }
        else
        {
            int32_t input_b_i_N_minus = input_b_i_N - N;
            accum[k].value[0] =
                (idx < input_b_i_N_minus) ? encoded_reg : (-encoded_reg);
            accum[k].value[1] = ((idx + blockDim.x) < input_b_i_N_minus)
                                    ? (-encoded_reg)
                                    : encoded_reg;
        }

        for (int i = 0; i < n; i++)
        {
            int32_t input_a_i = input_a[offset_lwe + i];
            int32_t input_a_i_N = torus_modulus_switch_log(input_a_i, N_power);

            Data64 offset_i =
                i * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);

            uint64_t2 accum3[4];
            for (int i2 = 0; i2 < (k + 1); i2++)
            {
                sdata32[idx] = accum[i2].value[0];
                sdata32[idx + blockDim.x] = accum[i2].value[1];
                __syncthreads();

                if (input_a_i_N < N)
                {
                    accum2[i2].value[0] =
                        (idx < input_a_i_N)
                            ? (-static_cast<int32_t>(
                                  sdata32[N - input_a_i_N + idx]))
                            : (static_cast<int32_t>(
                                  sdata32[idx - input_a_i_N]));
                    accum2[i2].value[1] =
                        ((idx + blockDim.x) < input_a_i_N)
                            ? (-static_cast<int32_t>(
                                  sdata32[N - input_a_i_N + idx]))
                            : (static_cast<int32_t>(
                                  sdata32[idx - input_a_i_N]));
                    accum2[i2].value[0] =
                        accum2[i2].value[0] - accum[i2].value[0];
                    accum2[i2].value[1] =
                        accum2[i2].value[1] - accum[i2].value[1];
                }
                else
                {
                    int32_t input_a_i_N_minus = input_a_i_N - N;
                    accum2[i2].value[0] =
                        (idx < input_a_i_N_minus)
                            ? (static_cast<int32_t>(
                                  sdata32[N - input_a_i_N_minus + idx]))
                            : (-static_cast<int32_t>(
                                  sdata32[idx - input_a_i_N_minus]));
                    accum2[i2].value[1] =
                        ((idx + blockDim.x) < input_a_i_N_minus)
                            ? (static_cast<int32_t>(
                                  sdata32[N - input_a_i_N_minus + idx]))
                            : (-static_cast<int32_t>(
                                  sdata32[idx - input_a_i_N_minus]));
                    accum2[i2].value[0] =
                        accum2[i2].value[0] - accum[i2].value[0];
                    accum2[i2].value[1] =
                        accum2[i2].value[1] - accum[i2].value[1];
                }

                Data64 offset_i2 = i2 * ((Data64) bk_length * (k + 1) * N);

                for (int i3 = 0; i3 < bk_length; i3++)
                {
                    Data64 offset_i3 =
                        (offset_i + offset_i2) + i3 * ((Data64) (k + 1) * N);

                    int shift = 32 - (bk_bit * (i3 + 1));
                    int32_t temp0 =
                        (((accum2[i2].value[0] + bk_offset) >> shift) &
                         bk_mask) -
                        bk_half;
                    int32_t temp1 =
                        (((accum2[i2].value[1] + bk_offset) >> shift) &
                         bk_mask) -
                        bk_half;

                    // PRE PROCESS
                    sdata64[idx] =
                        (temp0 <= 0)
                            ? static_cast<Data64>(modulus_reg.value + temp0)
                            : static_cast<Data64>(temp0);
                    sdata64[idx + blockDim.x] =
                        (temp1 <= 0)
                            ? static_cast<Data64>(modulus_reg.value + temp1)
                            : static_cast<Data64>(temp1);
                    __syncthreads();

                    SmallForwardNTT(sdata64, forward_root_of_unity_table,
                                    modulus_reg, false);

                    Data64 value0 = sdata64[idx];
                    Data64 value1 = sdata64[idx + blockDim.x];

                    for (int i4 = 0; i4 < (k + 1); i4++)
                    {
                        Data64 bk0 = boot_key[offset_i3 + (i4 * N) + idx];
                        Data64 bk1 =
                            boot_key[offset_i3 + (i4 * N) + idx + blockDim.x];

                        Data64 mul0 =
                            OPERATOR_GPU_64::mult(value0, bk0, modulus_reg);
                        Data64 mul1 =
                            OPERATOR_GPU_64::mult(value1, bk1, modulus_reg);

                        accum3[i4].value[0] = OPERATOR_GPU_64::add(
                            accum3[i4].value[0], mul0, modulus_reg);
                        accum3[i4].value[1] = OPERATOR_GPU_64::add(
                            accum3[i4].value[1], mul1, modulus_reg);
                    }
                }
            }

            for (int i4 = 0; i4 < (k + 1); i4++)
            {
                sdata64[idx] = accum3[i4].value[0];
                sdata64[idx + blockDim.x] = accum3[i4].value[1];
                __syncthreads();

                SmallInverseNTT(sdata64, inverse_root_of_unity_table,
                                modulus_reg, n_inverse_reg, false);

                accum3[i4].value[0] = sdata64[idx];
                accum3[i4].value[1] = sdata64[idx + blockDim.x];

                // POST PROCESS
                int32_t post_accum0 =
                    (accum3[i4].value[0] >= threshold)
                        ? static_cast<int32_t>(static_cast<int64_t>(
                              accum3[i4].value[0] - modulus_reg.value))
                        : static_cast<int32_t>(
                              static_cast<int64_t>(accum3[i4].value[0]));
                int32_t post_accum1 =
                    (accum3[i4].value[0] >= threshold)
                        ? static_cast<int32_t>(static_cast<int64_t>(
                              accum3[i4].value[0] - modulus_reg.value))
                        : static_cast<int32_t>(
                              static_cast<int64_t>(accum3[i4].value[0]));

                accum[i4].value[0] =
                    accum[i4].value[0] + static_cast<uint32_t>(post_accum0);
                accum[i4].value[1] =
                    accum[i4].value[1] + static_cast<uint32_t>(post_accum1);
            }
        }

        Data64 global_location =
            (Data64) block_x * (Data64) (k + 1) * (Data64) N;
        for (int i4 = 0; i4 < (k + 1); i4++)
        {
            output[global_location + (i4 * N) + idx] = accum[i4].value[0];
            output[global_location + (i4 * N) + idx + blockDim.x] =
                accum[i4].value[1];
        }
    }

    __global__ void tfhe_bootstrapping_kernel_unique_step1(
        const int32_t* input_a, const int32_t* input_b, Data64* output,
        const Data64* boot_key,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Modulus64 modulus, const int32_t encoded, const int32_t bk_offset,
        const int32_t bk_mask, const int32_t bk_half, int n, int N, int N_power,
        int k, int bk_bit, int bk_length)
    {
        __shared__ uint32_t shared_data32[1024];
        __shared__ Data64 shared_data64[1024];

        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size
        int block_y = blockIdx.y; // k
        int block_z = blockIdx.z; // l

        int32_t encoded_reg = encoded;
        const Modulus64 modulus_reg = modulus;

        int offset_lwe = block_x * n;

        Data64 offset_i = 0;
        Data64 offset_i2 = block_y * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_i3 =
            (offset_i + offset_i2) + (block_z * ((Data64) (k + 1) * N));

        Data64 offset_o =
            block_x * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_o2 = block_y * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_o3 =
            (offset_o + offset_o2) + (block_z * ((Data64) (k + 1) * N));

        int32_t input_b_reg = input_b[block_x];
        int32_t input_b_reg_N = torus_modulus_switch_log(input_b_reg, N_power);
        input_b_reg_N = static_cast<int32_t>(N << 1) - input_b_reg_N;

        int32_t2 temp;
        int32_t2 temp2;

        if (block_y == k)
        {
            if (input_b_reg_N < N)
            {
                temp.value[0] =
                    (idx_x < input_b_reg_N) ? (-encoded_reg) : encoded_reg;
                temp.value[1] = ((idx_x + blockDim.x) < input_b_reg_N)
                                    ? (-encoded_reg)
                                    : encoded_reg;
            }
            else
            {
                int32_t input_b_reg_N_minus = input_b_reg_N - N;
                temp.value[0] = (idx_x < input_b_reg_N_minus) ? encoded_reg
                                                              : (-encoded_reg);
                temp.value[1] = ((idx_x + blockDim.x) < input_b_reg_N_minus)
                                    ? encoded_reg
                                    : (-encoded_reg);
            }
        }

        int32_t input_a_reg = input_a[offset_lwe]; // + 0
        uint32_t input_a_reg_N = torus_modulus_switch_log(input_a_reg, N_power);

        shared_data32[idx_x] = temp.value[0];
        shared_data32[idx_x + blockDim.x] = temp.value[1];
        __syncthreads();

        if (input_a_reg_N < N)
        {
            temp2.value[0] =
                (idx_x < input_a_reg_N)
                    ? (-static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N + idx_x]))
                    : (static_cast<int32_t>(
                          shared_data32[idx_x - input_a_reg_N]));
            temp2.value[1] =
                ((idx_x + blockDim.x) < input_a_reg_N)
                    ? (-static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N +
                                        (idx_x + blockDim.x)]))
                    : (static_cast<int32_t>(
                          shared_data32[(idx_x + blockDim.x) - input_a_reg_N]));
            temp2.value[0] = temp2.value[0] - temp.value[0];
            temp2.value[1] = temp2.value[1] - temp.value[1];
        }
        else
        {
            int32_t input_a_reg_N_minus = input_a_reg_N - N;
            temp2.value[0] =
                (idx_x < input_a_reg_N_minus)
                    ? (static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N_minus + idx_x]))
                    : (-static_cast<int32_t>(
                          shared_data32[idx_x - input_a_reg_N_minus]));
            temp2.value[1] = ((idx_x + blockDim.x) < input_a_reg_N_minus)
                                 ? (static_cast<int32_t>(
                                       shared_data32[N - input_a_reg_N_minus +
                                                     (idx_x + blockDim.x)]))
                                 : (-static_cast<int32_t>(
                                       shared_data32[(idx_x + blockDim.x) -
                                                     input_a_reg_N_minus]));
            temp2.value[0] = temp2.value[0] - temp.value[0];
            temp2.value[1] = temp2.value[1] - temp.value[1];
        }

        int shift = 32 - (bk_bit * (block_z + 1));
        temp2.value[0] =
            (((temp2.value[0] + bk_offset) >> shift) & bk_mask) - bk_half;
        temp2.value[1] =
            (((temp2.value[1] + bk_offset) >> shift) & bk_mask) - bk_half;

        // PRE PROCESS
        shared_data64[idx_x] =
            (temp2.value[0] < 0)
                ? static_cast<Data64>(modulus_reg.value + temp2.value[0])
                : static_cast<Data64>(temp2.value[0]);
        shared_data64[idx_x + blockDim.x] =
            (temp2.value[1] < 0)
                ? static_cast<Data64>(modulus_reg.value + temp2.value[1])
                : static_cast<Data64>(temp2.value[1]);
        __syncthreads();

        SmallForwardNTT(shared_data64, forward_root_of_unity_table, modulus_reg,
                        false);

        Data64 ntt_value0 = shared_data64[idx_x];
        Data64 ntt_value1 = shared_data64[idx_x + blockDim.x];

#pragma unroll
        for (int i = 0; i < (k + 1); i++)
        {
            Data64 bk0 = boot_key[offset_i3 + (i * N) + idx_x];
            Data64 bk1 = boot_key[offset_i3 + (i * N) + idx_x + blockDim.x];

            Data64 mul0 = OPERATOR_GPU_64::mult(ntt_value0, bk0, modulus_reg);
            Data64 mul1 = OPERATOR_GPU_64::mult(ntt_value1, bk1, modulus_reg);

            output[offset_o3 + (i * N) + idx_x] = mul0;
            output[offset_o3 + (i * N) + idx_x + blockDim.x] = mul1;
        }
    }

    __global__ void tfhe_bootstrapping_kernel_regular_step1(
        const int32_t* input_a, const int32_t* input_b, const int32_t* input_c,
        Data64* output, const Data64* boot_key, int boot_index,
        const Root64* __restrict__ forward_root_of_unity_table,
        const Modulus64 modulus, const int32_t bk_offset, const int32_t bk_mask,
        const int32_t bk_half, int n, int N, int N_power, int k, int bk_bit,
        int bk_length)
    {
        __shared__ uint32_t shared_data32[1024];
        __shared__ Data64 shared_data64[1024];

        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size
        int block_y = blockIdx.y; // k
        int block_z = blockIdx.z; // l

        const Modulus64 modulus_reg = modulus;

        int offset_lwe = block_x * n;

        Data64 offset_i =
            boot_index * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_i2 = block_y * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_i3 =
            (offset_i + offset_i2) + (block_z * ((Data64) (k + 1) * N));

        Data64 offset_o =
            block_x * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_o2 = block_y * ((Data64) bk_length * (k + 1) * N);
        Data64 offset_o3 =
            (offset_o + offset_o2) + (block_z * ((Data64) (k + 1) * N));

        int32_t2 temp;
        int32_t2 temp2;

        int32_t input_a_reg = input_a[offset_lwe + boot_index];
        int32_t input_a_reg_N = torus_modulus_switch_log(input_a_reg, N_power);

        int offset_acc = block_x * (k + 1) * N;
        int offset_acc2 = block_y * N;
        offset_acc = offset_acc + offset_acc2 + idx_x;

        //
        temp.value[0] = input_c[offset_acc];
        temp.value[1] = input_c[offset_acc + blockDim.x];

        shared_data32[idx_x] = temp.value[0];
        shared_data32[idx_x + blockDim.x] = temp.value[1];
        __syncthreads();

        if (input_a_reg_N < N)
        {
            temp2.value[0] =
                (idx_x < input_a_reg_N)
                    ? (-static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N + idx_x]))
                    : (static_cast<int32_t>(
                          shared_data32[idx_x - input_a_reg_N]));
            temp2.value[1] =
                ((idx_x + blockDim.x) < input_a_reg_N)
                    ? (-static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N +
                                        (idx_x + blockDim.x)]))
                    : (static_cast<int32_t>(
                          shared_data32[(idx_x + blockDim.x) - input_a_reg_N]));
            temp2.value[0] = temp2.value[0] - temp.value[0];
            temp2.value[1] = temp2.value[1] - temp.value[1];
        }
        else
        {
            int32_t input_a_reg_N_minus = input_a_reg_N - N;
            temp2.value[0] =
                (idx_x < input_a_reg_N_minus)
                    ? (static_cast<int32_t>(
                          shared_data32[N - input_a_reg_N_minus + idx_x]))
                    : (-static_cast<int32_t>(
                          shared_data32[idx_x - input_a_reg_N_minus]));
            temp2.value[1] = ((idx_x + blockDim.x) < input_a_reg_N_minus)
                                 ? (static_cast<int32_t>(
                                       shared_data32[N - input_a_reg_N_minus +
                                                     (idx_x + blockDim.x)]))
                                 : (-static_cast<int32_t>(
                                       shared_data32[(idx_x + blockDim.x) -
                                                     input_a_reg_N_minus]));
            temp2.value[0] = temp2.value[0] - temp.value[0];
            temp2.value[1] = temp2.value[1] - temp.value[1];
        }

        int shift = 32 - (bk_bit * (block_z + 1));
        temp2.value[0] =
            (((temp2.value[0] + bk_offset) >> shift) & bk_mask) - bk_half;
        temp2.value[1] =
            (((temp2.value[1] + bk_offset) >> shift) & bk_mask) - bk_half;

        // PRE PROCESS
        shared_data64[idx_x] =
            (temp2.value[0] < 0)
                ? static_cast<Data64>(modulus_reg.value + temp2.value[0])
                : static_cast<Data64>(temp2.value[0]);
        shared_data64[idx_x + blockDim.x] =
            (temp2.value[1] < 0)
                ? static_cast<Data64>(modulus_reg.value + temp2.value[1])
                : static_cast<Data64>(temp2.value[1]);
        __syncthreads();

        SmallForwardNTT(shared_data64, forward_root_of_unity_table, modulus_reg,
                        false);

        Data64 ntt_value0 = shared_data64[idx_x];
        Data64 ntt_value1 = shared_data64[idx_x + blockDim.x];

#pragma unroll
        for (int i = 0; i < (k + 1); i++)
        {
            Data64 bk0 = boot_key[offset_i3 + (i * N) + idx_x];
            Data64 bk1 = boot_key[offset_i3 + (i * N) + idx_x + blockDim.x];

            Data64 mul0 = OPERATOR_GPU_64::mult(ntt_value0, bk0, modulus_reg);
            Data64 mul1 = OPERATOR_GPU_64::mult(ntt_value1, bk1, modulus_reg);

            output[offset_o3 + (i * N) + idx_x] = mul0;
            output[offset_o3 + (i * N) + idx_x + blockDim.x] = mul1;
        }
    }

    __global__ void tfhe_bootstrapping_kernel_unique_step2(
        const Data64* input, const int32_t* input_b, int32_t* output,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus,
        const int32_t encoded, int n, int N, int N_power, int k, int bk_length)
    {
        __shared__ Data64 shared_data64[1024];

        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size
        int block_y = blockIdx.y; // k

        Data64 offset_i =
            block_x * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);

        int32_t encoded_reg = encoded;
        const Modulus64 modulus_reg = modulus;
        const Data64 threshold = modulus_reg.value >> 1;
        const Ninverse64 n_inverse_reg = n_inverse;

        Data64 accum0 = 0ULL;
        Data64 accum1 = 0ULL;

        for (int i = 0; i < (k + 1); i++)
        {
            Data64 offset_i2 = i * ((Data64) bk_length * (k + 1) * N);

#pragma unroll
            for (int j = 0; j < bk_length; j++)
            {
                Data64 offset_i3 =
                    (offset_i + offset_i2) + (j * ((Data64) (k + 1) * N));
                offset_i3 = offset_i3 + (block_y * N);

                Data64 value0 = input[offset_i3 + idx_x];
                Data64 value1 = input[offset_i3 + idx_x + blockDim.x];

                accum0 = OPERATOR_GPU_64::add(accum0, value0, modulus_reg);
                accum1 = OPERATOR_GPU_64::add(accum1, value1, modulus_reg);
            }
        }

        shared_data64[idx_x] = accum0;
        shared_data64[idx_x + blockDim.x] = accum1;
        __syncthreads();

        SmallInverseNTT(shared_data64, inverse_root_of_unity_table, modulus_reg,
                        n_inverse_reg, false);

        accum0 = shared_data64[idx_x];
        accum1 = shared_data64[idx_x + blockDim.x];

        // POST PROCESS
        int32_t post_accum0 =
            (accum0 >= threshold)
                ? static_cast<int32_t>(
                      static_cast<int64_t>(accum0 - modulus_reg.value))
                : static_cast<int32_t>(static_cast<int64_t>(accum0));
        int32_t post_accum1 =
            (accum1 >= threshold)
                ? static_cast<int32_t>(
                      static_cast<int64_t>(accum1 - modulus_reg.value))
                : static_cast<int32_t>(static_cast<int64_t>(accum1));

        Data64 offset_o = block_x * (Data64) (k + 1) * N;
        offset_o = offset_o + (block_y * N);

        int32_t input_b_reg = input_b[block_x];
        int32_t input_b_reg_N = torus_modulus_switch_log(input_b_reg, N_power);
        input_b_reg_N = static_cast<int32_t>(N << 1) - input_b_reg_N;

        int32_t2 temp;

        if (block_y == k)
        {
            if (input_b_reg_N < N)
            {
                temp.value[0] =
                    (idx_x < input_b_reg_N) ? (-encoded_reg) : encoded_reg;
                temp.value[1] = ((idx_x + blockDim.x) < input_b_reg_N)
                                    ? (-encoded_reg)
                                    : encoded_reg;
            }
            else
            {
                int32_t input_b_reg_N_minus = input_b_reg_N - N;
                temp.value[0] = (idx_x < input_b_reg_N_minus) ? encoded_reg
                                                              : (-encoded_reg);
                temp.value[1] = ((idx_x + blockDim.x) < input_b_reg_N_minus)
                                    ? encoded_reg
                                    : (-encoded_reg);
            }

            post_accum0 = post_accum0 + temp.value[0];
            post_accum1 = post_accum1 + temp.value[1];
        }

        output[offset_o + idx_x] = post_accum0;
        output[offset_o + idx_x + blockDim.x] = post_accum1;
    }

    __global__ void tfhe_bootstrapping_kernel_regular_step2(
        const Data64* input, int32_t* output,
        const Root64* __restrict__ inverse_root_of_unity_table,
        const Ninverse64 n_inverse, const Modulus64 modulus, int n, int N,
        int k, int bk_length)
    {
        __shared__ Data64 shared_data64[1024];

        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size
        int block_y = blockIdx.y; // k

        Data64 offset_i =
            block_x * (Data64) (k + 1) * ((Data64) bk_length * (k + 1) * N);

        const Modulus64 modulus_reg = modulus;
        const Data64 threshold = modulus_reg.value >> 1;
        const Ninverse64 n_inverse_reg = n_inverse;

        Data64 accum0 = 0ULL;
        Data64 accum1 = 0ULL;

        for (int i = 0; i < (k + 1); i++)
        {
            Data64 offset_i2 = i * ((Data64) bk_length * (k + 1) * N);

#pragma unroll
            for (int j = 0; j < bk_length; j++)
            {
                Data64 offset_i3 =
                    (offset_i + offset_i2) + (j * ((Data64) (k + 1) * N));
                offset_i3 = offset_i3 + (block_y * N);

                Data64 value0 = input[offset_i3 + idx_x];
                Data64 value1 = input[offset_i3 + idx_x + blockDim.x];

                accum0 = OPERATOR_GPU_64::add(accum0, value0, modulus_reg);
                accum1 = OPERATOR_GPU_64::add(accum1, value1, modulus_reg);
            }
        }

        shared_data64[idx_x] = accum0;
        shared_data64[idx_x + blockDim.x] = accum1;
        __syncthreads();

        SmallInverseNTT(shared_data64, inverse_root_of_unity_table, modulus_reg,
                        n_inverse_reg, false);

        accum0 = shared_data64[idx_x];
        accum1 = shared_data64[idx_x + blockDim.x];

        // POST PROCESS
        int32_t post_accum0 =
            (accum0 >= threshold)
                ? static_cast<int32_t>(
                      static_cast<int64_t>(accum0 - modulus_reg.value))
                : static_cast<int32_t>(static_cast<int64_t>(accum0));
        int32_t post_accum1 =
            (accum1 >= threshold)
                ? static_cast<int32_t>(
                      static_cast<int64_t>(accum1 - modulus_reg.value))
                : static_cast<int32_t>(static_cast<int64_t>(accum1));

        Data64 offset_o = block_x * (Data64) (k + 1) * N;
        offset_o = offset_o + (block_y * N);

        output[offset_o + idx_x] = output[offset_o + idx_x] + post_accum0;
        output[offset_o + idx_x + blockDim.x] =
            output[offset_o + idx_x + blockDim.x] + post_accum1;
    }

    __global__ void tfhe_sample_extraction_kernel(const int32_t* input,
                                                  int32_t* output_a,
                                                  int32_t* output_b, int N,
                                                  int k, int index)
    {
        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size
        int block_y = blockIdx.y; // k

        Data64 offset_i = block_x * (Data64) (k + 1) * N;
        Data64 offset_i2 = offset_i + (block_y * N);

        int inner_index = index + 1;
        int idx_x2 = idx_x + blockDim.x;

        int32_t value0 = (idx_x < inner_index) ? input[offset_i2 + idx_x]
                                               : -input[offset_i2 + N - idx_x];
        int32_t value1 = (idx_x2 < inner_index)
                             ? input[offset_i2 + idx_x2]
                             : -input[offset_i2 + N - idx_x2];

        Data64 offset_o = block_x * (Data64) k * N;
        Data64 offset_o2 = offset_o + (block_y * N);

        output_a[offset_o2 + idx_x] = value0;
        output_a[offset_o2 + idx_x2] = value1;

        if ((idx_x == 0) && (block_y == 0))
        {
            Data64 offset_i3 = offset_i + (k * N);
            int32_t b_reg = input[offset_i3];

            output_b[block_x] = b_reg;
        }
    }

    // It will be more efficient!
    __global__ void tfhe_key_switching_kernel(
        const int32_t* input_a, const int32_t* input_b, int32_t* output_a,
        int32_t* output_b, const int32_t* ks_key_a, const int32_t* ks_key_b,
        int ks_base_bit_, int ks_length_, int n, int N, int k)
    {
        int idx_x = threadIdx.x;
        int block_x = blockIdx.x; // cipher size

        int ks_base_bit_reg = ks_base_bit_;
        int ks_length_reg = ks_length_;
        int n_reg = n;
        int N_reg = N;
        int k_reg = k;

        int base = 1 << ks_base_bit_reg;
        int precision_offset = 1
                               << (32 - (1 + ks_base_bit_reg * ks_length_reg));
        int mask = base - 1;

        int Nk_reg = N_reg * k_reg;
        int offset_i = block_x * Nk_reg;

        int32_t accum_a[2] = {0, 0};
        int32_t accum_b = 0;

        if (idx_x == 0)
        {
            accum_b = input_b[block_x];
        }

        for (int i = 0; i < Nk_reg; i++)
        {
            int32_t input_a_reg = input_a[offset_i + i];

            int offset_key_b_i = i * ks_length_reg * mask;
            int offset_key_a_i = offset_key_b_i * n_reg;

            for (int i2 = 0; i2 < ks_length_reg; i2++)
            {
                int32_t input_a_decomp =
                    (((input_a_reg + precision_offset) >>
                      (32 - ((i2 + 1) * ks_base_bit_reg))) &
                     mask) +
                    1;
                input_a_decomp = input_a_decomp - 1;
                int offset_key_b_i2 = offset_key_b_i + (i2 * mask);
                int offset_key_a_i3 = i2 * mask * n_reg;

                if (input_a_decomp != 0)
                {
                    int offset_key_a_i2 = (input_a_decomp - 1) * n_reg;
                    offset_key_a_i2 =
                        offset_key_a_i2 + offset_key_a_i + offset_key_a_i3;

                    int count = 0;
                    for (int i3 = idx_x; i3 < n; i3 += blockDim.x)
                    {
                        uint32_t ks_key_a_reg = ks_key_a[offset_key_a_i2 + i3];
                        accum_a[count] = accum_a[count] - ks_key_a_reg;
                        count++;
                    }

                    if (idx_x == 0)
                    {
                        uint32_t ks_key_b_reg =
                            ks_key_b[offset_key_b_i2 + (input_a_decomp - 1)];
                        accum_b = accum_b - ks_key_b_reg;
                    }
                }
            }
        }

        int offset_o = block_x * n_reg;

        int count = 0;
        for (int i = idx_x; i < n; i += blockDim.x)
        {
            output_a[offset_o + i] = accum_a[count];
            count++;
        }

        if (idx_x == 0)
        {
            output_b[block_x] = accum_b;
        }
    }

} // namespace heongpu