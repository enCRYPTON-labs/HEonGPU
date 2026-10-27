// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// The rescale modes of the CKKS context and the scales of the levels in the
// automatic mode.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Mode = heongpu::rescale_mode;

    heongpu::HEParams<S> params(int depth, Mode mode = Mode::automatic)
    {
        heongpu::HEParams<S> p;
        p.multiplicative_depth = depth;
        p.rescale_mode = mode;
        return p;
    }
} // namespace

TEST(RescaleMode, Manual_By_Default)
{
    auto context = params(3, Mode::manual).make_context();
    EXPECT_EQ(context->get_rescale_mode(), Mode::manual);
    EXPECT_THROW(context->get_level_scale(0), heongpu::StateError);
    context->generate();
    EXPECT_EQ(context->to_string().find("rescale"), std::string::npos);
}

TEST(RescaleMode, Needs_The_Moduli_And_A_Scale)
{
    auto empty = heongpu::GenHEContext<S>(heongpu::sec_level_type::sec128);
    EXPECT_THROW(empty->set_rescale_mode(Mode::automatic, std::pow(2.0, 40)),
                 heongpu::StateError);

    auto context = params(3, Mode::manual).make_context();
    for (double scale : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(context->set_rescale_mode(Mode::automatic, scale),
                     heongpu::ParameterError);
    }
    EXPECT_EQ(context->get_rescale_mode(), Mode::manual);
    EXPECT_THROW(context->set_rescale_mode(static_cast<Mode>(7)),
                 heongpu::ParameterError);
}

TEST(RescaleMode, Scales_Of_The_Levels)
{
    auto set = params(10).generate();
    EXPECT_EQ(set.rescale_mode, Mode::automatic);
    EXPECT_NE(set.to_string().find(", rescale=automatic)"), std::string::npos);

    auto context = set.make_context();
    ASSERT_EQ(context->get_rescale_mode(), Mode::automatic);
    EXPECT_EQ(context->get_level_scale(10), std::pow(2.0, 40));
    for (int level = 10; level > 0; level--)
    {
        double scale = context->get_level_scale(level);
        EXPECT_EQ(context->get_level_scale(level - 1),
                  scale * scale / double(set.coeff_modulus[level]));
        // HEParams keeps them next to the scale.
        EXPECT_NEAR(std::log2(context->get_level_scale(level - 1)), 40.0, 1e-4);
    }
    EXPECT_THROW(context->get_level_scale(-1), heongpu::ParameterError);
    EXPECT_THROW(context->get_level_scale(11), heongpu::ParameterError);

    // Back to manual.
    context->set_rescale_mode(Mode::manual);
    EXPECT_THROW(context->get_level_scale(10), heongpu::StateError);
}

TEST(RescaleMode, Not_Part_Of_The_Identity)
{
    auto manual = params(3, Mode::manual).make_context();
    auto automatic = params(3).make_context();
    manual->generate();
    automatic->generate();
    EXPECT_EQ(manual->context_id(), automatic->context_id());
    EXPECT_NE(automatic->to_string().find(", rescale=automatic, id="),
              std::string::npos);
}

TEST(RescaleMode, Products_Have_The_Scales_Of_The_Levels)
{
    // Multiplying two ciphertexts of a level and rescaling gives exactly
    // the scale of the next level.
    constexpr int depth = 6;
    auto context = params(depth).make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> x(context->get_poly_modulus_degree() / 2, 0.5);
    auto plaintext = encoder.encode(x, context->get_level_scale(depth));
    auto ciphertext = encryptor.encrypt(plaintext);
    for (int level = depth; level > 0; level--)
    {
        operators.multiply_inplace(ciphertext, ciphertext);
        operators.relinearize_inplace(ciphertext, relin_key);
        operators.rescale_inplace(ciphertext);
        ASSERT_EQ(ciphertext.level(), level - 1);
        EXPECT_EQ(ciphertext.scale(), context->get_level_scale(level - 1));
    }
}

TEST(RescaleMode, Encoding_With_The_Scales_Of_The_Levels)
{
    constexpr int depth = 5;
    auto manual = params(depth, Mode::manual).make_context();
    manual->generate();
    heongpu::HEEncoder<S> manual_encoder(manual);
    std::vector<double> x(manual->get_poly_modulus_degree() / 2, 0.25);
    EXPECT_THROW(manual_encoder.encode(x), heongpu::StateError);
    EXPECT_THROW(manual_encoder.encode(x, 2), heongpu::StateError);

    auto context = params(depth).make_context();
    context->generate();
    heongpu::HEEncoder<S> encoder(context);

    auto top = encoder.encode(x);
    EXPECT_EQ(top.depth(), 0);
    EXPECT_EQ(top.scale(), context->get_level_scale(depth));
    for (int d = 0; d <= depth; d++)
    {
        double scale = context->get_level_scale(depth - d);
        auto plaintext = encoder.encode(x, d);
        EXPECT_EQ(plaintext.depth(), d);
        EXPECT_EQ(plaintext.scale(), scale);
        // The same plaintext as with the scale given.
        auto reference = encoder.encode(x, scale, d);
        EXPECT_EQ(encoder.decode(plaintext), encoder.decode(reference));
    }

    std::vector<Complex64> z(x.size(), Complex64(0.5, -0.25));
    EXPECT_EQ(encoder.encode(z, 3).scale(), context->get_level_scale(2));

    heongpu::HostVector<double> host(x.size(), 0.25);
    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, host, 1);
    EXPECT_EQ(plaintext.scale(), context->get_level_scale(depth - 1));

    EXPECT_THROW(encoder.encode(x, depth + 1), heongpu::ParameterError);
    EXPECT_THROW(encoder.encode(x, -1), heongpu::ParameterError);
}
