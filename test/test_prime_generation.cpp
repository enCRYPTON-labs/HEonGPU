// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

// The primes generated for the parameter generator: the closest ones to a
// power of two, none skipped, and the excluded ones left out.

namespace
{
    int bit_length(Data64 value)
    {
        return 64 - __builtin_clzll(value);
    }

    // Expects every value 1 (mod factor) strictly between from and to, other
    // than the excluded ones, to be composite.
    void expect_no_prime_between(Data64 from, Data64 to, Data64 factor,
                                 const std::vector<Data64>& excluded = {})
    {
        if (from > to)
        {
            std::swap(from, to);
        }
        for (Data64 value = from + factor; value < to; value += factor)
        {
            if (std::find(excluded.begin(), excluded.end(), value) ==
                excluded.end())
            {
                EXPECT_FALSE(heongpu::is_prime(value)) << value;
            }
        }
    }
} // namespace

TEST(PrimeGeneration, Below)
{
    for (size_t n : {4096, 65536})
    {
        for (int bits : {30, 40, 60, 61})
        {
            Data64 factor = 2 * n;
            Data64 power = Data64(1) << bits;
            auto primes = heongpu::generate_primes_below(n, bits, 6);
            ASSERT_EQ(primes.size(), 6u);
            for (size_t i = 0; i < primes.size(); i++)
            {
                EXPECT_TRUE(heongpu::is_prime(primes[i]));
                EXPECT_EQ(primes[i] % factor, 1u);
                EXPECT_EQ(bit_length(primes[i]), bits);
                // Largest first, with no prime skipped.
                Data64 previous = (i == 0) ? power + 1 : primes[i - 1];
                EXPECT_LT(primes[i], previous);
                expect_no_prime_between(primes[i], previous, factor);
            }
        }
    }
}

TEST(PrimeGeneration, BelowMatchesGeneratePrimes)
{
    // generate_primes takes the primes of a size from the same list.
    auto expected = heongpu::generate_proper_primes(2 * 8192, 50, 4);
    EXPECT_EQ(heongpu::generate_primes_below(8192, 50, 4), expected);
}

TEST(PrimeGeneration, Around)
{
    for (size_t n : {4096, 65536})
    {
        for (int bits : {30, 40, 59, 60})
        {
            Data64 factor = 2 * n;
            Data64 power = Data64(1) << bits;
            auto primes = heongpu::generate_primes_around(n, bits, 7);
            ASSERT_EQ(primes.size(), 7u);

            std::vector<Data64> below, above;
            for (size_t i = 0; i < primes.size(); i++)
            {
                EXPECT_TRUE(heongpu::is_prime(primes[i]));
                EXPECT_EQ(primes[i] % factor, 1u);
                if (i % 2 == 0)
                {
                    EXPECT_LT(primes[i], power);
                    EXPECT_EQ(bit_length(primes[i]), bits);
                    below.push_back(primes[i]);
                }
                else
                {
                    EXPECT_GT(primes[i], power);
                    EXPECT_EQ(bit_length(primes[i]), bits + 1);
                    above.push_back(primes[i]);
                }
            }

            EXPECT_EQ(below, heongpu::generate_primes_below(n, bits, 4));
            for (size_t i = 0; i < above.size(); i++)
            {
                Data64 previous = (i == 0) ? power : above[i - 1];
                EXPECT_GT(above[i], previous);
                expect_no_prime_between(previous, above[i], factor);
            }
        }
    }
}

TEST(PrimeGeneration, AroundKeepsScale)
{
    // Rescaling by a prime below and then by one above leaves the scale
    // much closer to 2^40 than rescaling twice by primes below.
    auto around = heongpu::generate_primes_around(65536, 40, 2);
    auto below = heongpu::generate_primes_below(65536, 40, 2);
    double power = std::ldexp(1.0, 40);
    double around_drift = std::abs(
        std::log2(double(around[0]) / power * double(around[1]) / power));
    double below_drift = std::abs(
        std::log2(double(below[0]) / power * double(below[1]) / power));
    EXPECT_LT(around_drift, below_drift);
}

TEST(PrimeGeneration, Excluded)
{
    size_t n = 8192;
    Data64 factor = 2 * n;
    auto all = heongpu::generate_primes_around(n, 40, 6);
    std::vector<Data64> excluded = {all[0], all[3]};
    auto primes = heongpu::generate_primes_around(n, 40, 4, excluded);
    for (Data64 prime : primes)
    {
        EXPECT_EQ(std::find(excluded.begin(), excluded.end(), prime),
                  excluded.end());
    }
    EXPECT_EQ(primes[0], all[2]);
    EXPECT_EQ(primes[1], all[1]);
    EXPECT_EQ(primes[2], all[4]);
    EXPECT_EQ(primes[3], all[5]);

    auto top = heongpu::generate_primes_below(n, 60, 3);
    auto rest = heongpu::generate_primes_below(n, 60, 2, {top[0]});
    EXPECT_EQ(rest, std::vector<Data64>(top.begin() + 1, top.end()));
    expect_no_prime_between(rest[1], top[0], factor, {top[0], rest[0]});
}

TEST(PrimeGeneration, InvalidRequests)
{
    EXPECT_THROW(heongpu::generate_primes_below(8192, 29, 1),
                 heongpu::ParameterError);
    EXPECT_THROW(heongpu::generate_primes_below(8192, 62, 1),
                 heongpu::ParameterError);
    EXPECT_THROW(heongpu::generate_primes_around(8192, 61, 1),
                 heongpu::ParameterError);
    EXPECT_THROW(heongpu::generate_primes_below(6000, 40, 1),
                 heongpu::ParameterError);
    EXPECT_THROW(heongpu::generate_primes_around(2048, 40, 1),
                 heongpu::ParameterError);
    // There are only a few hundred primes 1 mod 2^17 between 2^29 and 2^30.
    EXPECT_THROW(heongpu::generate_primes_below(65536, 30, 100000),
                 heongpu::ParameterError);
}
