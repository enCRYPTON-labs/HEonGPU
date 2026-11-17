// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/serialization_payload.cuh>
#include <heongpu/util/hostvector.cuh>
#include <heongpu/util/util.cuh>
#include <openssl/crypto.h>
#include <algorithm>
#include <stdexcept>

namespace heongpu
{
    namespace detail
    {
        namespace
        {
            // Managed memory, as in the unified memory mode, and host memory
            // can be read and written by the host in place.
            bool host_accessible(const void* data)
            {
                cudaPointerAttributes attributes;
                if (cudaPointerGetAttributes(&attributes, data) != cudaSuccess)
                {
                    cudaGetLastError();
                    return false;
                }
                return attributes.type == cudaMemoryTypeManaged ||
                       attributes.type == cudaMemoryTypeHost;
            }

            HostVector<Data64> staging_buffer(std::size_t count,
                                              std::size_t chunk_values)
            {
                if (chunk_values == 0)
                {
                    throw std::invalid_argument(
                        "Payload chunk size must be > 0");
                }
                return HostVector<Data64>(std::min(count, chunk_values));
            }
        } // namespace

        void write_device_payload(std::ostream& os, const Data64* data,
                                  std::size_t count, cudaStream_t stream,
                                  bool sensitive, std::size_t chunk_values)
        {
            if (count == 0)
            {
                return;
            }
            if (host_accessible(data))
            {
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                os.write(reinterpret_cast<const char*>(data),
                         count * sizeof(Data64));
                return;
            }
            HostVector<Data64> staging = staging_buffer(count, chunk_values);
            for (std::size_t offset = 0; offset < count && os;)
            {
                const std::size_t values =
                    std::min(staging.size(), count - offset);
                const std::size_t bytes = values * sizeof(Data64);
                HEONGPU_CUDA_CHECK(
                    cudaMemcpyAsync(staging.data(), data + offset, bytes,
                                    cudaMemcpyDeviceToHost, stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                os.write(reinterpret_cast<const char*>(staging.data()), bytes);
                offset += values;
            }
            if (sensitive)
            {
                OPENSSL_cleanse(staging.data(),
                                staging.size() * sizeof(Data64));
            }
        }

        void read_device_payload(std::istream& is, Data64* data,
                                 std::size_t count, cudaStream_t stream,
                                 bool sensitive, std::size_t chunk_values)
        {
            if (count == 0)
            {
                return;
            }
            if (host_accessible(data))
            {
                // The memory may still be in use by work on stream that freed
                // it before it was allocated again.
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                is.read(reinterpret_cast<char*>(data), count * sizeof(Data64));
                return;
            }
            HostVector<Data64> staging = staging_buffer(count, chunk_values);
            for (std::size_t offset = 0; offset < count;)
            {
                const std::size_t values =
                    std::min(staging.size(), count - offset);
                is.read(reinterpret_cast<char*>(staging.data()),
                        values * sizeof(Data64));
                const std::size_t read =
                    static_cast<std::size_t>(is.gcount()) / sizeof(Data64);
                if (read != 0)
                {
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        data + offset, staging.data(), read * sizeof(Data64),
                        cudaMemcpyHostToDevice, stream));
                    // The next chunk reuses the staging buffer.
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                if (read != values)
                {
                    break;
                }
                offset += values;
            }
            if (sensitive)
            {
                OPENSSL_cleanse(staging.data(),
                                staging.size() * sizeof(Data64));
            }
        }
    } // namespace detail
} // namespace heongpu
