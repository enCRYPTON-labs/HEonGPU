// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_HOST_VECTOR_H
#define HEONGPU_HOST_VECTOR_H

#include <heongpu/util/memorypool.cuh>
#include <heongpu/util/devicevector.cuh>

namespace heongpu
{
    template <typename T> class DeviceVector;

    template <typename T>
    class HostVector : public std::vector<T, rmm_pinned_allocator<T>>
    {
      public:
        using Hvec = std::vector<T, rmm_pinned_allocator<T>>;
        using Hvec::vector;

        explicit HostVector(const DeviceVector<T>& ref,
                            cudaStream_t stream = cudaStreamDefault)
        {
            Hvec::resize(ref.size());
            cudaMemcpyAsync(Hvec::data(), ref.data(), ref.size() * sizeof(T),
                            cudaMemcpyDeviceToHost, stream);
        }
    };

} // namespace heongpu
#endif // HEONGPU_HOST_VECTOR_H
