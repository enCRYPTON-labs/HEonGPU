// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

// sum_slots and inner_product against sums computed on the CPU.

namespace
{
    constexpr int degree = 4096;
} // namespace

TEST(HEonGPU, CKKS_Sum_Slots_And_Inner_Product)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    constexpr int slots = degree / 2;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();

    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    auto galois_key = generator.galois_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> a(slots), b(slots);
    for (int i = 0; i < slots; i++)
    {
        a[i] = static_cast<double>(i % 32) / 1024.0;
        b[i] = static_cast<double>((7 * i) % 19) / 19.0;
    }
    const double scale = pow(2.0, 30);
    auto plaintext_a = encoder.encode(a, scale);
    auto plaintext_b = encoder.encode(b, scale);
    auto x = encryptor.encrypt(plaintext_a);
    auto y = encryptor.encrypt(plaintext_b);

    auto expect_all = [&](heongpu::Ciphertext<S>& ciphertext, double expected)
    {
        auto plaintext = decryptor.decrypt(ciphertext);
        auto values = encoder.decode(plaintext);
        for (int i = 0; i < slots; i++)
        {
            // The approximation error grows with the number of slots added.
            ASSERT_NEAR(values[i], expected, 1e-3 * expected + 1e-2)
                << "slot " << i;
        }
    };

    const double sum = std::accumulate(a.begin(), a.end(), 0.0);
    const double product =
        std::inner_product(a.begin(), a.end(), b.begin(), 0.0);

    auto summed = operators.sum_slots(x, galois_key);
    expect_all(summed, sum);

    auto inner = operators.inner_product(x, y, relin_key, galois_key);
    EXPECT_EQ(inner.depth(), 1);
    expect_all(inner, product);

    auto inner_plain = operators.inner_product(x, plaintext_b, galois_key);
    expect_all(inner_plain, product);
}

TEST(HEonGPU, BFV_Sum_Slots_And_Inner_Product)
{
    constexpr auto S = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36}, {37});
    context->set_plain_modulus(1032193);
    context->generate();

    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    // Keys for the steps 1, 2, 4, ..., 1024 of the rows of 2048 slots.
    heongpu::Galoiskey<S> galois_key(context, 10);
    generator.generate_galois_key(galois_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<uint64_t> a(degree), b(degree);
    for (int i = 0; i < degree; i++)
    {
        a[i] = i % 11;
        b[i] = (3 * i) % 7;
    }
    auto plaintext_a = encoder.encode(a);
    auto plaintext_b = encoder.encode(b);
    auto x = encryptor.encrypt(plaintext_a);
    auto y = encryptor.encrypt(plaintext_b);

    auto expect_all = [&](heongpu::Ciphertext<S>& ciphertext, uint64_t expected)
    {
        auto plaintext = decryptor.decrypt(ciphertext);
        auto values = encoder.decode(plaintext);
        for (int i = 0; i < degree; i++)
        {
            ASSERT_EQ(values[i], expected) << "slot " << i;
        }
    };

    const uint64_t sum = std::accumulate(a.begin(), a.end(), uint64_t(0));
    const uint64_t product =
        std::inner_product(a.begin(), a.end(), b.begin(), uint64_t(0));

    auto summed = operators.sum_slots(x, galois_key);
    expect_all(summed, sum);

    auto inner = operators.inner_product(x, y, relin_key, galois_key);
    expect_all(inner, product);

    auto inner_plain = operators.inner_product(x, plaintext_b, galois_key);
    expect_all(inner_plain, product);
}
