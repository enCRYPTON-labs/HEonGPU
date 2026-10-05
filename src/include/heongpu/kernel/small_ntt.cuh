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
    SmallForwardNTT(T* polynomial_in_shared, const Root<T>* root_of_unity_table,
                    const Modulus<T> modulus, bool reduction_poly_check);

    template <typename T>
    __device__ void
    SmallInverseNTT(T* polynomial_in_shared, const Root<T>* root_of_unity_table,
                    const Modulus<T> modulus, const Ninverse<T> n_inverse,
                    bool reduction_poly_check);

} // namespace heongpu

#endif // HEONGPU_SMALLNTT_H