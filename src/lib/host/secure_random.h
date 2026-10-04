// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HEONGPU_HOST_SECURE_RANDOM_H
#define HEONGPU_HOST_SECURE_RANDOM_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <openssl/crypto.h>
#include <openssl/rand.h>

namespace heongpu::detail
{
    inline void private_random_bytes(void* output, size_t size)
    {
        auto* bytes = static_cast<unsigned char*>(output);
        while (size)
        {
            const int count = static_cast<int>(std::min(
                size, static_cast<size_t>(std::numeric_limits<int>::max())));
            if (RAND_priv_bytes(bytes, count) != 1)
                throw std::runtime_error("RAND_priv_bytes failed");
            bytes += count;
            size -= count;
        }
    }

    // Host samples must not remain in freed allocator storage.
    template <typename T> struct PrivateRandomBuffer
    {
        explicit PrivateRandomBuffer(size_t size) : values(size) {}
        PrivateRandomBuffer(const PrivateRandomBuffer&) = delete;
        PrivateRandomBuffer& operator=(const PrivateRandomBuffer&) = delete;
        ~PrivateRandomBuffer()
        {
            if (!values.empty())
                OPENSSL_cleanse(values.data(), values.size() * sizeof(T));
        }
        std::vector<T> values;
    };

    // A private CSPRNG supplies unbiased standard integer/normal distributions.
    class SecretKeyRandomGenerator
    {
      public:
        using result_type = std::uint32_t;
        static constexpr result_type min() { return 0; }
        static constexpr result_type max()
        {
            return std::numeric_limits<result_type>::max();
        }
        SecretKeyRandomGenerator() = default;
        SecretKeyRandomGenerator(const SecretKeyRandomGenerator&) = delete;
        SecretKeyRandomGenerator&
        operator=(const SecretKeyRandomGenerator&) = delete;
        ~SecretKeyRandomGenerator()
        {
            OPENSSL_cleanse(buffer_.data(), sizeof(buffer_));
        }
        result_type operator()()
        {
            if (next_ == buffer_.size())
            {
                private_random_bytes(buffer_.data(), sizeof(buffer_));
                next_ = 0;
            }
            return buffer_[next_++];
        }

      private:
        std::array<result_type, 256> buffer_{};
        size_t next_ = buffer_.size();
    };

    inline void sample_ternary_positions(int degree, int weight,
                                         std::vector<int>& positions,
                                         std::vector<int>& values)
    {
        PrivateRandomBuffer<int> indices(degree);
        for (int i = 0; i < degree; ++i)
            indices.values[i] = i;
        SecretKeyRandomGenerator random;
        std::uniform_int_distribution<int> sign(0, 1);
        for (int i = 0; i < weight; ++i)
        {
            std::uniform_int_distribution<int> position(i, degree - 1);
            std::swap(indices.values[i], indices.values[position(random)]);
            positions[i] = indices.values[i];
            values[i] = sign(random) ? 1 : -1;
        }
    }

    inline int32_t torus32(double value)
    {
        // Round in a signed type before reducing modulo 2^32. Casting a
        // negative floating point sample directly to uint32_t is undefined
        // behavior.
        const auto rounded =
            std::llround((value - std::trunc(value)) * 4294967296.0);
        return static_cast<int32_t>(static_cast<uint32_t>(rounded));
    }
} // namespace heongpu::detail
#endif
