// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// The CKKS parameter generator: the parameters never exceed the security
// limit, take the smallest ring size and the largest P, and a computation of
// the requested depth uses every level with the expected precision.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Params = heongpu::HEParams<S>;
    using Level = heongpu::sec_level_type;

    int bit_length(Data64 value)
    {
        return 64 - __builtin_clzll(value);
    }

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

    Params params(int depth, int scale_bits = 40, Level level = Level::sec128)
    {
        Params p;
        p.multiplicative_depth = depth;
        p.scale_bit_size = scale_bits;
        p.security_level = level;
        return p;
    }
} // namespace

TEST(CKKSParameters, Never_Exceed_The_Security_Limit)
{
    for (Level level : {Level::sec128, Level::sec192, Level::sec256})
    {
        for (int scale_bits : {30, 35, 40, 45, 50, 55, 59})
        {
            int fitted = 0;
            for (int depth = 0; depth <= 60; depth++)
            {
                Params p = params(depth, scale_bits, level);
                heongpu::HEParameterSet<S> set;
                try
                {
                    set = p.generate();
                }
                catch (const heongpu::ParameterError&)
                {
                    // Deeper computations do not fit either.
                    EXPECT_THROW(
                        params(depth + 1, scale_bits, level).generate(),
                        heongpu::ParameterError);
                    break;
                }
                fitted++;
                SCOPED_TRACE(set.to_string());
                std::size_t n = set.poly_modulus_degree;
                EXPECT_LE(set.total_bit_count(), set.max_bit_count());
                EXPECT_EQ(set.multiplicative_depth(), depth);

                // The first prime has 60 bits, the scaling primes are next to
                // the scale and the P primes have 60 bits.
                EXPECT_EQ(bit_length(set.coeff_modulus[0]), 60);
                for (int i = 1; i <= depth; i++)
                {
                    int bits = bit_length(set.coeff_modulus[i]);
                    EXPECT_TRUE((bits == scale_bits) ||
                                (bits == scale_bits + 1));
                }
                for (Data64 prime : set.special_modulus)
                {
                    EXPECT_EQ(bit_length(prime), 60);
                }

                // The smallest ring size, and as many P primes as fit.
                if (n > 4096)
                {
                    Params smaller = p;
                    smaller.poly_modulus_degree = n / 2;
                    EXPECT_THROW(smaller.generate(), heongpu::ParameterError);
                }
                EXPECT_GE(set.special_modulus.size(), 1u);
                EXPECT_LE(set.special_modulus.size(), size_t(depth + 1));
                if (set.special_modulus.size() < size_t(depth + 1))
                {
                    EXPECT_GT(set.total_bit_count() + 60, set.max_bit_count());
                }

                // Every prime is distinct and 1 mod 2N.
                std::vector<Data64> all = set.coeff_modulus;
                all.insert(all.end(), set.special_modulus.begin(),
                           set.special_modulus.end());
                for (size_t i = 0; i < all.size(); i++)
                {
                    EXPECT_EQ(all[i] % (2 * n), 1u);
                    for (size_t j = 0; j < i; j++)
                    {
                        EXPECT_NE(all[i], all[j]);
                    }
                }

                // The context accepts them.
                EXPECT_NO_THROW(set.make_context());
            }
            EXPECT_GT(fitted, 0);
        }
    }
}

TEST(CKKSParameters, Choices)
{
    struct Expected
    {
        int depth, scale_bits;
        std::size_t n;
        std::size_t special_count;
        int digits;
    };
    for (const auto& e : std::vector<Expected>{{1, 40, 8192, 1, 2},
                                               {2, 40, 8192, 1, 3},
                                               {5, 40, 16384, 2, 3},
                                               {10, 40, 32768, 6, 2},
                                               {10, 50, 32768, 5, 3},
                                               {15, 40, 32768, 3, 6},
                                               {20, 40, 65536, 14, 2},
                                               {30, 40, 65536, 8, 4}})
    {
        auto set = params(e.depth, e.scale_bits).generate();
        SCOPED_TRACE(set.to_string());
        EXPECT_EQ(set.poly_modulus_degree, e.n);
        EXPECT_EQ(set.special_modulus.size(), e.special_count);
        EXPECT_EQ(set.digit_count(), e.digits);
    }

    // The first prime and the P primes have 60 bits, the scaling primes 40
    // or 41.
    auto set = params(5).generate();
    EXPECT_EQ(set.to_string(),
              "HEParameterSet<CKKS>(N=16384, Q=60+40+41+41+40+41 bits, "
              "P=60+60 bits, total=383/438 bits, digits=3, scale=2^40, "
              "security=sec128)");
    EXPECT_EQ(set.scale, std::pow(2.0, 40));
}

TEST(CKKSParameters, Scale_Stays_Around_The_Scale_When_Squaring)
{
    // Squaring and rescaling from the top level: s_{l-1} = s_l^2 / q_l.
    for (int scale_bits : {40, 50})
    {
        for (int depth : {10, 20, 30})
        {
            auto set = params(depth, scale_bits).generate();
            SCOPED_TRACE(set.to_string());
            double scale = std::ldexp(1.0, scale_bits);
            for (int level = depth; level >= 1; level--)
            {
                scale = scale * scale / double(set.coeff_modulus[level]);
                ASSERT_NEAR(std::log2(scale), scale_bits, 1e-3) << level;
            }
        }
    }
}

TEST(CKKSParameters, Given_Ring_Size_And_Special_Count)
{
    Params p = params(5);
    p.poly_modulus_degree = 65536;
    auto set = p.generate();
    EXPECT_EQ(set.poly_modulus_degree, 65536u);
    EXPECT_EQ(set.special_modulus.size(), 6u);

    p.special_modulus_count = 2;
    EXPECT_EQ(p.generate().special_modulus.size(), 2u);

    // Two P primes need a larger ring than one.
    Params q = params(2);
    EXPECT_EQ(q.generate().poly_modulus_degree, 8192u);
    q.special_modulus_count = 2;
    EXPECT_EQ(q.generate().poly_modulus_degree, 16384u);

    Params small = params(5);
    small.poly_modulus_degree = 8192;
    EXPECT_EQ(message_of([&] { small.generate(); }),
              "Depth 5 with the scale 2^40 does not fit sec128 with N = 8192: "
              "Q and one P prime need 323 bits, the limit is 218. At most "
              "depth 2 fits.");

    small.poly_modulus_degree = 4096;
    small.special_modulus_count = 2;
    EXPECT_EQ(message_of([&] { small.generate(); }),
              "Depth 5 with the scale 2^40 does not fit sec128 with N = 4096: "
              "Q and 2 P primes need 382 bits, the limit is 109. No depth "
              "fits.");
}

TEST(CKKSParameters, Too_Deep)
{
    EXPECT_EQ(message_of([] { params(41).generate(); }),
              "Depth 41 with the scale 2^40 does not fit sec128 with any N up "
              "to 65536: Q and one P prime need 1780 bits, the limit is 1761. "
              "At most depth 40 fits.");
    EXPECT_NO_THROW(params(40).generate());
}

TEST(CKKSParameters, Invalid_Requests)
{
    EXPECT_THROW(params(-1).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 29).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 60).generate(), heongpu::ParameterError);
    EXPECT_THROW(params(3, 40, Level::none).generate(),
                 heongpu::ParameterError);

    Params p = params(3, 50);
    p.first_modulus_bit_size = 49;
    EXPECT_THROW(p.generate(), heongpu::ParameterError);
    p.first_modulus_bit_size = 61;
    EXPECT_THROW(p.generate(), heongpu::ParameterError);

    Params q = params(3);
    q.poly_modulus_degree = 10000;
    EXPECT_THROW(q.generate(), heongpu::ParameterError);
    q.poly_modulus_degree = 131072;
    EXPECT_THROW(q.generate(), heongpu::ParameterError);
    q.poly_modulus_degree = 0;
    q.special_modulus_count = -1;
    EXPECT_THROW(q.generate(), heongpu::ParameterError);
}

TEST(CKKSParameters, Apply_Checks_The_Security_Level)
{
    auto set = params(3).generate();
    auto context = heongpu::GenHEContext<S>(Level::sec192);
    EXPECT_THROW(set.apply(context), heongpu::ParameterError);

    auto same = heongpu::GenHEContext<S>(Level::sec128);
    set.apply(same);
    EXPECT_EQ(same->get_poly_modulus_degree(), 16384);
    EXPECT_EQ(same->get_ciphertext_modulus_count(), 4);
    EXPECT_EQ(same->get_key_modulus_count(), 8);
}

TEST(CKKSParameters, Squaring_On_The_GPU_Keeps_The_Scale)
{
    constexpr int depth = 12;
    Params p = params(depth);
    auto context = p.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    // Signs, whose squares are 1.
    int slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> x(slots);
    for (int i = 0; i < slots; i++)
    {
        x[i] = (i % 3 == 0) ? -1.0 : 1.0;
    }
    auto plaintext = encoder.encode(x, p.scale());
    auto ciphertext = encryptor.encrypt(plaintext);
    for (int level = 0; level < depth; level++)
    {
        operators.multiply_inplace(ciphertext, ciphertext);
        operators.relinearize_inplace(ciphertext, relin_key);
        operators.rescale_inplace(ciphertext);
        ASSERT_NEAR(std::log2(ciphertext.scale()), 40.0, 1e-3) << level;
    }
    EXPECT_EQ(ciphertext.level(), 0);

    auto decrypted = decryptor.decrypt(ciphertext);
    auto values = encoder.decode(decrypted);
    for (int i = 0; i < slots; i++)
    {
        // Squaring doubles the relative error at every level.
        ASSERT_NEAR(values[i], 1.0, std::ldexp(1.0, -8)) << i;
    }
}

// Multiplies by a ciphertext of signs at every level, so the values keep
// their size and the error grows slowly, down to level 0.
class CKKSParametersCircuit
    : public ::testing::TestWithParam<std::pair<int, int>>
{
};

TEST_P(CKKSParametersCircuit, Uses_Every_Level)
{
    auto [depth, scale_bits] = GetParam();
    Params p = params(depth, scale_bits);
    auto set = p.generate();
    SCOPED_TRACE(set.to_string());

    auto context = p.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    int slots = context->get_poly_modulus_degree() / 2;
    std::mt19937 random(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<double> x(slots), signs(slots);
    for (int i = 0; i < slots; i++)
    {
        x[i] = uniform(random);
        signs[i] = (uniform(random) < 0) ? -1.0 : 1.0;
    }

    auto x_plaintext = encoder.encode(x, p.scale());
    auto signs_plaintext = encoder.encode(signs, p.scale());
    auto ciphertext = encryptor.encrypt(x_plaintext);
    auto sign_ciphertext = encryptor.encrypt(signs_plaintext);
    std::vector<double> expected = x;
    for (int level = 0; level < depth; level++)
    {
        operators.multiply_inplace(ciphertext, sign_ciphertext);
        operators.relinearize_inplace(ciphertext, relin_key);
        operators.rescale_inplace(ciphertext);
        operators.mod_drop_inplace(sign_ciphertext);
        for (int i = 0; i < slots; i++)
        {
            expected[i] *= signs[i];
        }
    }
    EXPECT_EQ(ciphertext.level(), 0);

    // The scale stays around 2^scale_bits.
    EXPECT_LT(std::abs(std::log2(ciphertext.scale()) - scale_bits), 1e-4);

    auto plaintext = decryptor.decrypt(ciphertext);
    auto values = encoder.decode(plaintext);
    double error = 0.0;
    for (int i = 0; i < slots; i++)
    {
        error = std::max(error, std::abs(values[i] - expected[i]));
    }
    // At least 19 bits of precision with a 2^40 scale after 20 levels, more
    // with larger scales.
    EXPECT_LT(error, std::ldexp(1.0, 21 - scale_bits));
}

INSTANTIATE_TEST_SUITE_P(Depths, CKKSParametersCircuit,
                         ::testing::Values(std::make_pair(1, 40),
                                           std::make_pair(5, 40),
                                           std::make_pair(10, 40),
                                           std::make_pair(10, 50),
                                           std::make_pair(20, 40)));
