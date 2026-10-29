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

// TFHE MUX takes its control input from host memory too.
TEST(Inputs, TFHE_MUX_With_A_Control_Input_In_Host_Memory)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = heongpu::GenHEContext<S>();
    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Bootstrappingkey<S> boot_key(context);
    keygen.generate_bootstrapping_key(boot_key, secret_key);
    heongpu::HEEncryptor<S> encryptor(context, secret_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context);

    constexpr int size = 32;
    std::vector<bool> a(size), b(size), c(size), expected(size);
    for (int i = 0; i < size; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
        c[i] = (i / 4) % 2;
        expected[i] = c[i] ? a[i] : b[i];
    }
    heongpu::Ciphertext<S> x(context), y(context), control(context);
    encryptor.encrypt(x, a);
    encryptor.encrypt(y, b);
    encryptor.encrypt(control, c);
    control.store_in_host();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    heongpu::Ciphertext<S> output(context);
    logic.MUX(x, y, control, output, boot_key);
    std::vector<bool> result;
    decryptor.decrypt(output, result);
    EXPECT_EQ(result, expected);
}
