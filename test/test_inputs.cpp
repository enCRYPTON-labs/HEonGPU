// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// Operations leave their inputs unchanged.

namespace
{
    constexpr int degree = 8192;

    template <typename T> std::string bytes(const T& object)
    {
        std::stringstream stream;
        object.save(stream);
        return stream.str();
    }
} // namespace

TEST(Inputs, BFV_NOT_Leaves_Its_Input)
{
    constexpr auto S = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<S>();
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_default_values(1);
    context->set_plain_modulus(1032193);
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context, encoder);

    std::vector<uint64_t> message(degree);
    for (int i = 0; i < degree; i++)
    {
        message[i] = i % 2;
    }
    auto plain = encoder.encode(message);
    auto input = encryptor.encrypt(plain);
    const std::string before = bytes(input);

    heongpu::Ciphertext<S> output(context);
    logic.NOT(input, output);
    EXPECT_EQ(bytes(input), before);

    auto decrypted = decryptor.decrypt(output);
    std::vector<uint64_t> result;
    encoder.decode(result, decrypted);
    for (int i = 0; i < degree; i++)
    {
        ASSERT_EQ(result[i], 1 - message[i]) << i;
    }
}

TEST(Inputs, CKKS_NOT_Leaves_Its_Input)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    auto context = heongpu::GenHEContext<S>();
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    context->generate();
    const double scale = std::pow(2.0, 30);
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context, encoder, scale);

    std::vector<double> message(degree / 2);
    for (int i = 0; i < degree / 2; i++)
    {
        message[i] = i % 2;
    }
    auto plain = encoder.encode(message, scale);
    auto input = encryptor.encrypt(plain);
    const std::string before = bytes(input);

    heongpu::Ciphertext<S> output(context);
    logic.NOT(input, output);
    EXPECT_EQ(bytes(input), before);

    auto decrypted = decryptor.decrypt(output);
    std::vector<double> result;
    encoder.decode(result, decrypted);
    for (int i = 0; i < degree / 2; i++)
    {
        ASSERT_NEAR(result[i], 1 - message[i], 1e-3) << i;
    }
}
