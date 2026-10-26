// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#ifdef HEONGPU_TEST_WRAP_RAND_BYTES
namespace
{
    // Buffers filled by RAND_bytes, recorded while `recording` is set.
    bool recording = false;
    std::vector<std::pair<const unsigned char*, int>> requests;
} // namespace

extern "C" int __real_RAND_bytes(unsigned char*, int);
extern "C" int __wrap_RAND_bytes(unsigned char* buffer, int size)
{
    if (recording)
    {
        requests.emplace_back(buffer, size);
    }
    return __real_RAND_bytes(buffer, size);
}

namespace
{
    template <heongpu::Scheme S> void expect_entropy_and_nonce()
    {
        auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({36, 36}, {37});
        if constexpr (S == heongpu::Scheme::BFV)
        {
            context->set_plain_modulus(1032193);
        }

        requests.clear();
        recording = true;
        context->generate();
        recording = false;

        // The random number generator is seeded with 16 bytes of entropy
        // and an 8 byte nonce, each read into its own buffer.
        ASSERT_EQ(requests.size(), 2u);
        EXPECT_EQ(requests[0].second, 16);
        EXPECT_EQ(requests[1].second, 8);
        EXPECT_NE(requests[0].first, requests[1].first);
    }
} // namespace

TEST(HEonGPU, CKKS_Context_Seeds_Entropy_And_Nonce)
{
    expect_entropy_and_nonce<heongpu::Scheme::CKKS>();
}

TEST(HEonGPU, BFV_Context_Seeds_Entropy_And_Nonce)
{
    expect_entropy_and_nonce<heongpu::Scheme::BFV>();
}
#endif
