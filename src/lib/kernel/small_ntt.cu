// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/small_ntt.cuh>

namespace heongpu
{
    template <typename T>
    __device__ void
    SmallForwardNTT(T* polynomial_in_shared, const Root<T>* root_of_unity_table,
                    const Modulus<T> modulus, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int N_power = 10;

        int t_2 = N_power - 1;
        int t_ = 9;
        int m = 1;
        int t = 1 << t_;

        int in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        int current_root_index;

#pragma unroll
        for (int lp = 0; lp < 4; lp++)
        {
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (idx_x >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (idx_x >> t_2);
            }

            gpuntt::CooleyTukeyUnit(polynomial_in_shared[in_shared_address],
                                    polynomial_in_shared[in_shared_address + t],
                                    root_of_unity_table[current_root_index],
                                    modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((idx_x >> t_) << t_) + idx_x;
            __syncthreads();
        }

#pragma unroll
        for (int lp = 0; lp < 6; lp++)
        {
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (idx_x >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (idx_x >> t_2);
            }
            gpuntt::CooleyTukeyUnit(polynomial_in_shared[in_shared_address],
                                    polynomial_in_shared[in_shared_address + t],
                                    root_of_unity_table[current_root_index],
                                    modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            __syncwarp();
            if (lp < 5)
                in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        }
        __syncthreads();
    }

    template <typename T>
    __device__ void
    SmallInverseNTT(T* polynomial_in_shared, const Root<T>* root_of_unity_table,
                    const Modulus<T> modulus, const Ninverse<T> n_inverse,
                    bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;

        int t_2 = 0;
        int t_ = 0;
        int m = 1 << 9;
        int t = 1 << t_;

        int in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        int current_root_index;
#pragma unroll
        for (int lp = 0; lp < 10; lp++)
        {
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (idx_x >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (idx_x >> t_2);
            }

            gpuntt::GentlemanSandeUnit(
                polynomial_in_shared[in_shared_address],
                polynomial_in_shared[in_shared_address + t],
                root_of_unity_table[current_root_index], modulus);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address = ((idx_x >> t_) << t_) + idx_x;

            __syncthreads();
        }

        polynomial_in_shared[idx_x] = OPERATOR_GPU<T>::mult(
            polynomial_in_shared[idx_x], n_inverse, modulus);
        polynomial_in_shared[idx_x + blockDim.x] = OPERATOR_GPU<T>::mult(
            polynomial_in_shared[idx_x + blockDim.x], n_inverse, modulus);

        __syncthreads();
    }

    template __device__ void SmallForwardNTT<Data32>(
        Data32* polynomial_in_shared, const Root<Data32>* root_of_unity_table,
        const Modulus<Data32> modulus, bool reduction_poly_check);

    template __device__ void SmallForwardNTT<Data64>(
        Data64* polynomial_in_shared, const Root<Data64>* root_of_unity_table,
        const Modulus<Data64> modulus, bool reduction_poly_check);

    template __device__ void SmallInverseNTT<Data32>(
        Data32* polynomial_in_shared, const Root<Data32>* root_of_unity_table,
        const Modulus<Data32> modulus, const Ninverse<Data32> n_inverse,
        bool reduction_poly_check);

    template __device__ void SmallInverseNTT<Data64>(
        Data64* polynomial_in_shared, const Root<Data64>* root_of_unity_table,
        const Modulus<Data64> modulus, const Ninverse<Data64> n_inverse,
        bool reduction_poly_check);

} // namespace heongpu