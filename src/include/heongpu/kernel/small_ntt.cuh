// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SMALLNTT_H
#define HEONGPU_SMALLNTT_H

#include "gpuntt/ntt_merge/ntt.cuh"

namespace heongpu
{
    template <typename T>
    __device__ void
    SmallForwardNTT(T* polynomial_in_shared, const T* root_of_unity_table,
                    const Modulus<T> modulus, bool reduction_poly_check);

    template <typename T>
    __device__ void SmallInverseNTT(T* polynomial_in_shared,
                                    const T* root_of_unity_table,
                                    const Modulus<T> modulus, const T n_inverse,
                                    bool reduction_poly_check);

// The definitions live in the header, not in a translation unit of their own,
// because the AMD GPU build compiles without relocatable device code and a
// __device__ function called from another translation unit does not link there.
// Only a device compiler may see them: this header is reached from heongpu.hpp,
// so the host compiler parses it as well and must find the declarations alone.
// Turning relocatable device code on for AMD instead (-fgpu-rdc, the
// counterpart of the CUDA_SEPARABLE_COMPILATION the NVIDIA build sets) would
// let the definitions stay in a file of their own, and the device link still
// inlines the calls, but it leaves no complete device image in the archive: the
// device link then happens wherever the library is linked, so every consumer
// has to enable the HIP language and link through the HIP compiler driver.
// Keeping the definitions here asks nothing of consumers.
#if defined(__CUDACC__) || defined(__HIPCC__)

    template <typename T>
    __device__ inline void
    SmallForwardNTT(T* polynomial_in_shared, const T* root_of_unity_table,
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
    __device__ inline void
    SmallInverseNTT(T* polynomial_in_shared, const T* root_of_unity_table,
                    const Modulus<T> modulus, const T n_inverse,
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

#endif // __CUDACC__ || __HIPCC__

} // namespace heongpu

#endif // HEONGPU_SMALLNTT_H