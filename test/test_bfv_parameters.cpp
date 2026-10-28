// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

// The BFV parameter generator: the parameters never exceed the security
// limit and take the smallest ring size, and a computation of the requested
// depth decrypts correctly with about the noise margin left.

namespace
{
    constexpr auto S = heongpu::Scheme::BFV;
    using Params = heongpu::HEParams<S>;
    using Level = heongpu::sec_level_type;

    // Runs function, expects it to throw ParameterError and returns the
    // message.
    template <typename F> std::string message_of(F function)
    {
        try
        {
            function();
        }
        catch (const heongpu::ParameterError& error)
        {
            return error.what();
        }
        ADD_FAILURE() << "no ParameterError";
        return "";
    }

    int bit_length(Data64 value)
    {
        return 64 - __builtin_clzll(value);
    }

    Params params(int depth, int plain_bits = 20, Level level = Level::sec128)
    {
        Params p;
        p.multiplicative_depth = depth;
        p.plain_modulus_bit_size = plain_bits;
        p.security_level = level;
        return p;
    }
} // namespace

TEST(BFVParameters, Never_Exceed_The_Security_Limit)
{
    for (Level level : {Level::sec128, Level::sec192, Level::sec256})
    {
        for (int plain_bits : {17, 20, 25, 30})
        {
            int fitted = 0;
            for (int depth = 0; depth <= 40; depth++)
            {
                Params p = params(depth, plain_bits, level);
                heongpu::HEParameterSet<S> set;
                try
                {
                    set = p.generate();
                }
                catch (const heongpu::ParameterError&)
                {
                    // Deeper computations do not fit either.
                    EXPECT_THROW(
                        params(depth + 1, plain_bits, level).generate(),
                        heongpu::ParameterError);
                    break;
                }
                fitted++;
                SCOPED_TRACE(set.to_string());
                std::size_t n = set.poly_modulus_degree;
                EXPECT_LE(set.total_bit_count(), set.max_bit_count());

                // P primes with a bit more than the largest Q prime, at most
                // 60, distinct from the Q primes and t.
                int q_count = int(set.coeff_modulus.size());
                int p_count = int(set.special_modulus.size());
                int p_bits = std::min(bit_length(set.coeff_modulus[0]) + 1, 60);
                for (Data64 prime : set.special_modulus)
                {
                    EXPECT_EQ(bit_length(prime), p_bits);
                    EXPECT_EQ(prime % (2 * n), 1u);
                    EXPECT_NE(prime, set.plain_modulus);
                    for (Data64 q : set.coeff_modulus)
                    {
                        EXPECT_NE(prime, q);
                    }
                }

                // As many P primes as fit, at most one per Q prime, and the
                // fewest that give the same number of digits.
                int digits = set.digit_count();
                EXPECT_GE(p_count, 1);
                EXPECT_LE(p_count, q_count);
                EXPECT_EQ(p_count, (q_count + digits - 1) / digits);
                if (digits > 1)
                {
                    int fewer = (q_count + digits - 2) / (digits - 1);
                    EXPECT_GT(set.total_bit_count() +
                                  (fewer - p_count) * p_bits,
                              set.max_bit_count());
                }

                // A batching plaintext modulus of the requested size.
                EXPECT_TRUE(heongpu::is_prime(set.plain_modulus));
                EXPECT_EQ(set.plain_modulus % (2 * n), 1u);
                EXPECT_EQ(64 - __builtin_clzll(set.plain_modulus), plain_bits);

                // The smallest ring size.
                if (n > 4096)
                {
                    Params smaller = p;
                    smaller.poly_modulus_degree = n / 2;
                    EXPECT_THROW(smaller.generate(), heongpu::ParameterError);
                }

                EXPECT_NO_THROW(set.make_context());
            }
            EXPECT_GT(fitted, 0);
        }
    }
}

TEST(BFVParameters, Choices)
{
    struct Expected
    {
        int depth, plain_bits;
        std::size_t n;
        std::size_t q_count;
        Data64 t;
        std::size_t special_count;
        int digits;
    };
    for (const auto& e :
         std::vector<Expected>{{0, 20, 4096, 1, 1032193, 1, 1},
                               {2, 20, 8192, 2, 1032193, 2, 1},
                               {4, 20, 16384, 3, 786433, 3, 1},
                               {10, 20, 16384, 7, 786433, 1, 7},
                               {20, 20, 32768, 13, 786433, 2, 7},
                               {30, 20, 65536, 19, 786433, 10, 2},
                               {4, 17, 8192, 3, 114689, 1, 3},
                               {5, 17, 16384, 4, 65537, 4, 1},
                               {1, 30, 8192, 2, 1073692673, 2, 1},
                               {8, 30, 32768, 7, 1073479681, 7, 1}})
    {
        auto set = params(e.depth, e.plain_bits).generate();
        SCOPED_TRACE(set.to_string());
        EXPECT_EQ(set.poly_modulus_degree, e.n);
        EXPECT_EQ(set.coeff_modulus.size(), e.q_count);
        EXPECT_EQ(set.plain_modulus, e.t);
        EXPECT_EQ(set.special_modulus.size(), e.special_count);
        EXPECT_EQ(set.digit_count(), e.digits);
    }

    // Q primes whose bit sizes differ by at most one, and a P prime with one
    // more bit than the largest.
    EXPECT_EQ(params(1).generate().to_string(),
              "HEParameterSet<BFV>(N=4096, Q=36+35 bits, P=37 bits, "
              "total=108/109 bits, digits=2, t=1032193, security=sec128)");
    // As many P primes as fit, one digit.
    EXPECT_EQ(params(4).generate().to_string(),
              "HEParameterSet<BFV>(N=16384, Q=59+58+58 bits, P=60+60+60 bits, "
              "total=355/438 bits, digits=1, t=786433, security=sec128)");
}

TEST(BFVParameters, Given_Plaintext_Modulus_And_Ring_Size)
{
    Params p = params(3);
    p.plain_modulus = 65537;
    auto set = p.generate();
    EXPECT_EQ(set.plain_modulus, 65537u);

    p.poly_modulus_degree = 32768;
    set = p.generate();
    EXPECT_EQ(set.poly_modulus_degree, 32768u);
    EXPECT_EQ(set.plain_modulus, 65537u);

    // 17 bit batching primes exist up to N = 2^15.
    Params small = params(1, 17);
    small.poly_modulus_degree = 65536;
    EXPECT_EQ(message_of([&] { small.generate(); }),
              "No prime 1 mod 131072 has 17 bits! Batching with N = 65536 "
              "needs a plaintext modulus of at least 18 bits.");
}

TEST(BFVParameters, Too_Deep)
{
    EXPECT_EQ(message_of([] { params(60).generate(); }),
              "Depth 60 with the plaintext modulus 786433 does not fit sec128 "
              "with any N up to 65536: Q and one P prime need 2265 bits, the "
              "limit is 1761. At most depth 46 fits.");
    EXPECT_NO_THROW(params(46).generate());

    // Depth 30 needs N = 2^16, where batching needs 18 bits.
    EXPECT_EQ(message_of([] { params(30, 17).generate(); }),
              "Depth 30 does not fit sec128 with N up to 32768, and no prime 1 "
              "mod 131072 has 17 bits! Batching with N = 65536 needs a "
              "plaintext modulus of at least 18 bits.");
}

TEST(BFVParameters, Invalid_Requests)
{
    EXPECT_THROW(params(-1).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 13).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 31).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 20, Level::none).generate(),
                 heongpu::ParameterError);

    Params p = params(3);
    p.plain_modulus = 1;
    EXPECT_THROW(p.generate(), heongpu::ParameterError);
    p.plain_modulus = Data64(1) << 31;
    EXPECT_THROW(p.generate(), heongpu::ParameterError);

    Params q = params(3);
    q.poly_modulus_degree = 10000;
    EXPECT_THROW(q.generate(), heongpu::ParameterError);
}

TEST(BFVParameters, Apply_Checks_The_Security_Level)
{
    auto set = params(3).generate();
    auto context = heongpu::GenHEContext<S>(Level::sec256);
    EXPECT_THROW(set.apply(context), heongpu::ParameterError);
}

// Squares a ciphertext depth times, the noise growth of depth multiplications
// of fresh ciphertexts.
class BFVParametersCircuit
    : public ::testing::TestWithParam<std::pair<int, int>>
{
};

TEST_P(BFVParametersCircuit, Leaves_The_Margin)
{
    auto [depth, plain_bits] = GetParam();
    Params p = params(depth, plain_bits);
    auto set = p.generate();
    SCOPED_TRACE(set.to_string());

    auto context = set.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<S> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::Relinkey<S> relin_key(context);
    generator.generate_relin_key(relin_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::size_t n = set.poly_modulus_degree;
    Data64 t = set.plain_modulus;
    std::mt19937_64 random(7);
    std::vector<uint64_t> message(n);
    for (auto& value : message)
    {
        value = random() % t;
    }

    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, message);
    heongpu::Ciphertext<S> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);
    std::vector<uint64_t> expected = message;
    for (int level = 0; level < depth; level++)
    {
        operators.multiply_inplace(ciphertext, ciphertext);
        operators.relinearize_inplace(ciphertext, relin_key);
        for (auto& value : expected)
        {
            value = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(value) * value) % t);
        }
    }

    int budget = decryptor.remainder_noise_budget(ciphertext);
    // The margin of 10 bits, and about one bit per multiplication the model
    // overestimates.
    EXPECT_GE(budget, 10);
    EXPECT_LE(budget, 13 + depth);

    heongpu::Plaintext<S> result(context);
    decryptor.decrypt(result, ciphertext);
    std::vector<uint64_t> values;
    encoder.decode(values, result);
    EXPECT_EQ(values, expected);
}

INSTANTIATE_TEST_SUITE_P(
    Depths, BFVParametersCircuit,
    ::testing::Values(std::make_pair(0, 20), std::make_pair(1, 20),
                      std::make_pair(2, 17), std::make_pair(4, 20),
                      std::make_pair(6, 25), std::make_pair(10, 20),
                      std::make_pair(8, 30), std::make_pair(20, 20)));
