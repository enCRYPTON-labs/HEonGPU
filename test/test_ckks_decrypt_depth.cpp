// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

// A plaintext decrypted from a ciphertext that has lost moduli has the size of
// its remaining moduli, so that saving or copying it reads only its own data.

TEST(HEonGPU, CKKS_Decrypted_Plaintext_Has_The_Size_Of_Its_Depth)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;
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

    std::vector<double> message(degree / 2, 0.75);
    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, message, pow(2.0, 30));
    heongpu::Ciphertext<S> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);
    operators.mod_drop_inplace(ciphertext);

    heongpu::Plaintext<S> decrypted(context);
    decryptor.decrypt(decrypted, ciphertext);
    EXPECT_EQ(decrypted.depth(), 1);
    EXPECT_EQ(decrypted.size(), degree * 2);

    // Saving and loading it keeps the values.
    std::stringstream stream;
    decrypted.save(stream);
    heongpu::Plaintext<S> loaded;
    loaded.load(stream);
    std::vector<double> values;
    encoder.decode(values, loaded);
    for (int i = 0; i < degree / 2; i++)
    {
        ASSERT_NEAR(values[i], 0.75, 1e-3) << "slot " << i;
    }
}
