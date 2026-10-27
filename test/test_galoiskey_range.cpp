// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// The rotation range of a Galois key is chosen at run time with max_shift;
// MAX_SHIFT in kernel/defines.h is only the default.

namespace
{
    constexpr int degree = 4096;
    constexpr int row_size = degree / 2;

    // Beyond the range (-255, 255) of the default keys (MAX_SHIFT = 8).
    constexpr int large_shift = 300;
} // namespace

TEST(HEonGPU, BFV_Galoiskey_Range_Is_Chosen_At_Run_Time)
{
    constexpr auto S = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36}, {37});
    context->set_plain_modulus(1032193);
    context->generate();

    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<S> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<uint64_t> message(degree);
    for (int i = 0; i < degree; i++)
    {
        message[i] = i;
    }
    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, message);
    heongpu::Ciphertext<S> input(context);
    encryptor.encrypt(input, plaintext);

    heongpu::Galoiskey<S> default_key(context);
    generator.generate_galois_key(default_key, secret_key);
    heongpu::Ciphertext<S> output(context);
    EXPECT_THROW(operators.rotate_rows(input, output, default_key, large_shift),
                 heongpu::KeyNotFoundError);

    heongpu::Galoiskey<S> wide_key(context, 9);
    generator.generate_galois_key(wide_key, secret_key);
    operators.rotate_rows(input, output, wide_key, large_shift);
    heongpu::Plaintext<S> result_plaintext(context);
    decryptor.decrypt(result_plaintext, output);
    std::vector<uint64_t> result;
    encoder.decode(result, result_plaintext);
    for (int i = 0; i < degree; i++)
    {
        const int row = i / row_size;
        const int expected =
            row * row_size + (i % row_size + large_shift) % row_size;
        ASSERT_EQ(result[i], message[expected]) << "slot " << i;
    }

    EXPECT_THROW(heongpu::Galoiskey<S>(context, -1), heongpu::ParameterError);
}

TEST(HEonGPU, CKKS_Galoiskey_Range_Is_Chosen_At_Run_Time)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();

    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<S> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> message(row_size);
    for (int i = 0; i < row_size; i++)
    {
        message[i] = static_cast<double>(i % 101) / 101.0;
    }
    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, message, pow(2.0, 30));
    heongpu::Ciphertext<S> input(context);
    encryptor.encrypt(input, plaintext);

    heongpu::Galoiskey<S> wide_key(context, 9);
    generator.generate_galois_key(wide_key, secret_key);
    heongpu::Ciphertext<S> output(context);
    operators.rotate(input, output, wide_key, large_shift);
    heongpu::Plaintext<S> result_plaintext(context);
    decryptor.decrypt(result_plaintext, output);
    std::vector<double> result;
    encoder.decode(result, result_plaintext);
    for (int i = 0; i < row_size; i++)
    {
        ASSERT_NEAR(result[i], message[(i + large_shift) % row_size], 1e-2)
            << "slot " << i;
    }

    EXPECT_THROW(heongpu::Galoiskey<S>(context, -1), heongpu::ParameterError);
}
