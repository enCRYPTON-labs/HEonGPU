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
        using typename Hvec::value_type;
        using typename Hvec::allocator_type;
        using typename Hvec::size_type;
        using typename Hvec::difference_type;
        using typename Hvec::reference;
        using typename Hvec::const_reference;
        using typename Hvec::pointer;
        using typename Hvec::const_pointer;
        using typename Hvec::iterator;
        using typename Hvec::const_iterator;

        // Inherit constructors
        using Hvec::Hvec;

        // Default constructor
        HostVector() = default;

        // Copy constructor
        HostVector(const HostVector& other) : Hvec(other) {}

        // Move constructor
        HostVector(HostVector&& other) noexcept : Hvec(std::move(other)) {}

        // Copy assignment operator
        HostVector& operator=(const HostVector& other)
        {
            if (this != &other)
            {
                Hvec::operator=(other);
            }
            return *this;
        }

        // Move assignment operator
        HostVector& operator=(HostVector&& other) noexcept
        {
            if (this != &other)
            {
                Hvec::operator=(std::move(other));
            }
            return *this;
        }

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
