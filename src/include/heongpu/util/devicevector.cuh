// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_DEVICE_VECTOR_H
#define HEONGPU_DEVICE_VECTOR_H

#include <heongpu/util/memorypool.cuh>
#include <heongpu/util/hostvector.cuh>
#include "gpufft/fft.cuh" // for COMPLEX define

namespace heongpu
{
    template <typename T> class HostVector;

    template <typename T> class DeviceVector : public rmm::device_uvector<T>
    {
        using Dvec = rmm::device_uvector<T>;
        using Source = rmm::mr::device_memory_resource*;

      public:
        explicit DeviceVector(size_t size = 0,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(size, stream, memory_resource)
        {
        }

        explicit DeviceVector(const DeviceVector& other,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(other, stream, memory_resource)
        {
        }

        DeviceVector& operator=(const DeviceVector& other)
        {
            if (this != &other)
            {
                Dvec::resize(0, other.stream());
                Dvec::resize(other.size(), other.stream());
                cudaMemcpyAsync(Dvec::data(), other.data(),
                                other.size() * sizeof(T),
                                cudaMemcpyDeviceToDevice, other.stream());
            }
            return *this;
        }

        DeviceVector(DeviceVector&& assign) noexcept : Dvec(std::move(assign))
        {
        }

        DeviceVector& operator=(DeviceVector&& assign) noexcept
        {
            if (this != &assign)
            {
                Dvec::operator=(std::move(assign));
            }
            return *this;
        }

        explicit DeviceVector(const HostVector<T>& ref,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * sizeof(T),
                            cudaMemcpyHostToDevice, stream);
        }

        explicit DeviceVector(const std::vector<T>& ref,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * sizeof(T),
                            cudaMemcpyHostToDevice, stream);
        }

        // for complex forced // TODO: make it better way.
        explicit DeviceVector(const std::vector<Complex64>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        explicit DeviceVector(const HostVector<Complex64>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        // for uint64_t forced // TODO: make it better way.
        explicit DeviceVector(const std::vector<uint64_t>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        explicit DeviceVector(const HostVector<uint64_t>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        // for int64_t forced // TODO: make it better way.
        explicit DeviceVector(const std::vector<int64_t>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        explicit DeviceVector(const HostVector<int64_t>& ref, size_t d_size,
                              cudaStream_t stream = cudaStreamDefault,
                              Source memory_resource =
                                  MemoryPool::instance().get_device_resource())
            : Dvec(ref.size(), stream, memory_resource)
        {
            cudaMemcpyAsync(Dvec::data(), ref.data(), ref.size() * d_size,
                            cudaMemcpyHostToDevice, stream);
        }

        void resize(size_t size, cudaStream_t stream = cudaStreamDefault,
                    Source memory_resource =
                        MemoryPool::instance().get_device_resource())
        {
            Dvec::resize(size, stream);
        }

        void reserve(size_t size, cudaStream_t stream = cudaStreamDefault,
                     Source memory_resource =
                         MemoryPool::instance().get_device_resource())
        {
            Dvec::reserve(size, stream);
        }

        void append(const DeviceVector& out,
                    cudaStream_t stream = cudaStreamDefault,
                    Source memory_resource =
                        MemoryPool::instance().get_device_resource())
        {
            size_t original_size = Dvec::size();
            resize(original_size + out.size(), stream, memory_resource);
            cudaMemcpyAsync(Dvec::data() + original_size, out.data(),
                            out.size() * sizeof(T), cudaMemcpyDeviceToDevice,
                            stream);
        }
    };

} // namespace heongpu
#endif // HEONGPU_DEVICE_VECTOR_H