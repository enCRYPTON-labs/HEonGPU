// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// The value-returning versions give the results of the versions with an
// output parameter.

namespace
{
    using heongpu::CudaException;
    constexpr int degree = 4096;

    template <typename T> std::vector<Data64> data(T& object)
    {
        std::vector<Data64> values;
        object.get_data(values);
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        return values;
    }

    template <heongpu::Scheme S>
    std::vector<Data64> data(heongpu::Plaintext<S>& plaintext)
    {
        std::vector<Data64> values(plaintext.size());
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        HEONGPU_CUDA_CHECK(cudaMemcpy(values.data(), plaintext.data(),
                                      values.size() * sizeof(Data64),
                                      cudaMemcpyDeviceToHost));
        return values;
    }
} // namespace

TEST(HEonGPU, CKKS_Value_Returning_Versions)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Ciphertext = heongpu::Ciphertext<S>;
    using Plaintext = heongpu::Plaintext<S>;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();

    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    auto galois_key = generator.galois_key(secret_key);
    std::vector<int> shifts = {3};
    auto shift_key = generator.galois_key(secret_key, shifts);

    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);
    const double scale = pow(2.0, 30);
    std::vector<double> message(degree / 2);
    for (int i = 0; i < degree / 2; i++)
    {
        message[i] = static_cast<double>(i % 11) / 11.0;
    }

    auto plaintext = encoder.encode(message, scale);
    Plaintext plaintext_reference(context);
    encoder.encode(plaintext_reference, message, scale);
    EXPECT_EQ(data(plaintext), data(plaintext_reference));

    auto deep_plaintext = encoder.encode(message, scale, 1);
    Plaintext deep_reference(context);
    encoder.encode(deep_reference, message, scale, 1);
    EXPECT_EQ(data(deep_plaintext), data(deep_reference));

    // Keys and encryption are random: check that they work.
    auto x = encryptor.encrypt(plaintext);
    auto y = encryptor.encrypt(plaintext);
    auto product = operators.multiply(x, y);
    operators.relinearize_inplace(product, relin_key);
    operators.rescale_inplace(product);
    auto decrypted = decryptor.decrypt(product);
    auto values = encoder.decode(decrypted);
    for (int i = 0; i < degree / 2; i++)
    {
        ASSERT_NEAR(values[i], message[i] * message[i], 1e-2) << i;
    }

    // Deterministic functions give the same data.
    Plaintext decrypted_reference(context);
    decryptor.decrypt(decrypted_reference, product);
    EXPECT_EQ(data(decrypted), data(decrypted_reference));
    std::vector<double> values_reference;
    encoder.decode(values_reference, decrypted);
    EXPECT_EQ(values, values_reference);

    auto same = [&](Ciphertext result, auto reference_operation)
    {
        Ciphertext reference(context);
        reference_operation(reference);
        EXPECT_EQ(data(result), data(reference));
    };
    same(operators.add(x, y), [&](Ciphertext& r) { operators.add(x, y, r); });
    same(operators.sub(x, y), [&](Ciphertext& r) { operators.sub(x, y, r); });
    same(operators.multiply(x, y),
         [&](Ciphertext& r) { operators.multiply(x, y, r); });
    same(operators.negate(x), [&](Ciphertext& r) { operators.negate(x, r); });
    same(operators.add_plain(x, plaintext),
         [&](Ciphertext& r) { operators.add_plain(x, plaintext, r); });
    same(operators.sub_plain(x, plaintext),
         [&](Ciphertext& r) { operators.sub_plain(x, plaintext, r); });
    same(operators.multiply_plain(x, plaintext),
         [&](Ciphertext& r) { operators.multiply_plain(x, plaintext, r); });
    same(operators.add_plain(x, 0.25),
         [&](Ciphertext& r) { operators.add_plain(x, 0.25, r); });
    same(operators.sub_plain(x, 0.25),
         [&](Ciphertext& r) { operators.sub_plain(x, 0.25, r); });
    same(operators.multiply_plain(x, 0.25, scale),
         [&](Ciphertext& r) { operators.multiply_plain(x, 0.25, r, scale); });
    same(operators.rotate_rows(x, galois_key, 5),
         [&](Ciphertext& r) { operators.rotate_rows(x, r, galois_key, 5); });
    same(operators.rotate_rows(x, shift_key, 3),
         [&](Ciphertext& r) { operators.rotate_rows(x, r, shift_key, 3); });
    same(operators.conjugate(x, galois_key),
         [&](Ciphertext& r) { operators.conjugate(x, r, galois_key); });
    same(operators.mod_drop(x),
         [&](Ciphertext& r) { operators.mod_drop(x, r); });

    auto dropped = operators.mod_drop(plaintext);
    Plaintext dropped_reference(context);
    operators.mod_drop(plaintext, dropped_reference);
    EXPECT_EQ(data(dropped), data(dropped_reference));
}

TEST(HEonGPU, BFV_Value_Returning_Versions)
{
    constexpr auto S = heongpu::Scheme::BFV;
    using Ciphertext = heongpu::Ciphertext<S>;
    using Plaintext = heongpu::Plaintext<S>;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36}, {37});
    context->set_plain_modulus(1032193);
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
    std::vector<uint64_t> message(degree);
    for (int i = 0; i < degree; i++)
    {
        message[i] = i % 17;
    }

    auto plaintext = encoder.encode(message);
    Plaintext plaintext_reference(context);
    encoder.encode(plaintext_reference, message);
    EXPECT_EQ(data(plaintext), data(plaintext_reference));
    std::vector<int64_t> signed_message(degree, -3);
    auto signed_plaintext = encoder.encode(signed_message);
    Plaintext signed_reference(context);
    encoder.encode(signed_reference, signed_message);
    EXPECT_EQ(data(signed_plaintext), data(signed_reference));

    auto x = encryptor.encrypt(plaintext);
    auto y = encryptor.encrypt(plaintext);
    auto product = operators.multiply(x, y);
    operators.relinearize_inplace(product, relin_key);
    auto decrypted = decryptor.decrypt(product);
    auto values = encoder.decode(decrypted);
    for (int i = 0; i < degree; i++)
    {
        ASSERT_EQ(values[i], message[i] * message[i]) << i;
    }

    Plaintext decrypted_reference(context);
    decryptor.decrypt(decrypted_reference, product);
    EXPECT_EQ(data(decrypted), data(decrypted_reference));

    auto same = [&](Ciphertext result, auto reference_operation)
    {
        Ciphertext reference(context);
        reference_operation(reference);
        EXPECT_EQ(data(result), data(reference));
    };
    same(operators.add(x, y), [&](Ciphertext& r) { operators.add(x, y, r); });
    same(operators.sub(x, y), [&](Ciphertext& r) { operators.sub(x, y, r); });
    same(operators.multiply(x, y),
         [&](Ciphertext& r) { operators.multiply(x, y, r); });
    same(operators.negate(x), [&](Ciphertext& r) { operators.negate(x, r); });
    same(operators.add_plain(x, plaintext),
         [&](Ciphertext& r) { operators.add_plain(x, plaintext, r); });
    same(operators.sub_plain(x, plaintext),
         [&](Ciphertext& r) { operators.sub_plain(x, plaintext, r); });
    same(operators.multiply_plain(x, plaintext),
         [&](Ciphertext& r) { operators.multiply_plain(x, plaintext, r); });
    same(operators.rotate_rows(x, galois_key, 5),
         [&](Ciphertext& r) { operators.rotate_rows(x, r, galois_key, 5); });
    same(operators.rotate_columns(x, galois_key),
         [&](Ciphertext& r) { operators.rotate_columns(x, r, galois_key); });
}
