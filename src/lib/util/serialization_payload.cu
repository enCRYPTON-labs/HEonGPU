// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/serialization_payload.cuh>
#include <heongpu/util/hostvector.cuh>
#include <heongpu/util/util.cuh>
#include <openssl/crypto.h>

namespace heongpu
{
    namespace detail
    {
        void write_device_payload(std::ostream& os, const Data64* data,
                                  std::size_t count, cudaStream_t stream,
                                  bool sensitive)
        {
            if (count == 0)
            {
                return;
            }
            const std::size_t bytes = count * sizeof(Data64);
            HostVector<Data64> staging(count);
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(staging.data(), data, bytes,
                                               cudaMemcpyDeviceToHost, stream));
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            os.write(reinterpret_cast<const char*>(staging.data()), bytes);
            if (sensitive)
            {
                OPENSSL_cleanse(staging.data(), bytes);
            }
        }

        void read_device_payload(std::istream& is, Data64* data,
                                 std::size_t count, cudaStream_t stream,
                                 bool sensitive)
        {
            if (count == 0)
            {
                return;
            }
            const std::size_t bytes = count * sizeof(Data64);
            HostVector<Data64> staging(count);
            is.read(reinterpret_cast<char*>(staging.data()), bytes);
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(data, staging.data(), bytes,
                                               cudaMemcpyHostToDevice, stream));
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            if (sensitive)
            {
                OPENSSL_cleanse(staging.data(), bytes);
            }
        }
    } // namespace detail
} // namespace heongpu
