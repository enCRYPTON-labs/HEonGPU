// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// HEEncoder::encode with a depth gives the plaintext that encoding and then
// mod_drop give.

namespace
{
    using heongpu::CudaException;
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;
    constexpr int slots = degree / 2;

    std::vector<Data64> device_data(heongpu::Plaintext<scheme>& plaintext)
    {
        std::vector<Data64> values(plaintext.size());
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        HEONGPU_CUDA_CHECK(cudaMemcpy(values.data(), plaintext.data(),
                                      values.size() * sizeof(Data64),
                                      cudaMemcpyDeviceToHost));
        return values;
    }

    class CKKSEncodeDepth : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            context =
                heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
            context->generate();
            encoder = std::make_unique<heongpu::HEEncoder<scheme>>(context);
            operators = std::make_unique<heongpu::HEArithmeticOperator<scheme>>(
                context, *encoder);
            for (int i = 0; i < slots; i++)
            {
                message[i] = static_cast<double>(i % 13) / 13.0;
            }
        }

        const double scale = pow(2.0, 30);
        std::vector<double> message = std::vector<double>(slots);
        heongpu::HEContext<scheme> context;
        std::unique_ptr<heongpu::HEEncoder<scheme>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<scheme>> operators;
    };
} // namespace

TEST_F(CKKSEncodeDepth, Same_As_Encode_And_Mod_Drop)
{
    for (int depth = 0; depth < 3; depth++)
    {
        SCOPED_TRACE(depth);
        heongpu::Plaintext<scheme> direct(context);
        encoder->encode(direct, message, scale, depth);

        heongpu::Plaintext<scheme> dropped(context);
        encoder->encode(dropped, message, scale);
        for (int i = 0; i < depth; i++)
        {
            heongpu::Plaintext<scheme> next(context);
            operators->mod_drop(dropped, next);
            dropped = std::move(next);
        }

        EXPECT_EQ(direct.depth(), depth);
        EXPECT_EQ(direct.size(), degree * (3 - depth));
        EXPECT_EQ(direct.depth(), dropped.depth());
        EXPECT_EQ(direct.size(), dropped.size());
        EXPECT_EQ(direct.scale(), dropped.scale());
        EXPECT_EQ(device_data(direct), device_data(dropped));
    }
}

TEST_F(CKKSEncodeDepth, Plaintext_Matches_A_Ciphertext_Of_That_Depth)
{
    heongpu::HEKeyGenerator<scheme> generator(context);
    heongpu::Secretkey<scheme> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<scheme> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncryptor<scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<scheme> decryptor(context, secret_key);

    heongpu::Plaintext<scheme> plaintext(context);
    encoder->encode(plaintext, message, scale);
    heongpu::Ciphertext<scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);
    operators->mod_drop_inplace(ciphertext);
    ASSERT_EQ(ciphertext.depth(), 1);

    // Stored on the host, like any plaintext can be.
    heongpu::Plaintext<scheme> addend(context);
    encoder->encode(addend, std::vector<double>(slots, 0.5), scale,
                    ciphertext.depth(),
                    heongpu::ExecutionOptions().set_storage_type(
                        heongpu::storage_type::host));
    EXPECT_FALSE(addend.is_on_device());
    operators->add_plain_inplace(ciphertext, addend);

    heongpu::Plaintext<scheme> result_plaintext(context);
    decryptor.decrypt(result_plaintext, ciphertext);
    std::vector<double> result;
    encoder->decode(result, result_plaintext);
    for (int i = 0; i < slots; i++)
    {
        ASSERT_NEAR(result[i], message[i] + 0.5, 1e-3) << "slot " << i;
    }
}

TEST_F(CKKSEncodeDepth, Encrypts_At_The_Depth_Of_The_Plaintext)
{
    heongpu::HEKeyGenerator<scheme> generator(context);
    heongpu::Secretkey<scheme> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<scheme> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncryptor<scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<scheme> decryptor(context, secret_key);

    for (int depth = 0; depth < 3; depth++)
    {
        SCOPED_TRACE(depth);
        heongpu::Plaintext<scheme> plaintext(context);
        encoder->encode(plaintext, message, scale, depth);
        heongpu::Ciphertext<scheme> ciphertext(context);
        encryptor.encrypt(ciphertext, plaintext);
        EXPECT_EQ(ciphertext.depth(), depth);
        EXPECT_EQ(ciphertext.scale(), scale);

        // The same as encrypting at the top and dropping the moduli.
        heongpu::Plaintext<scheme> top(context);
        encoder->encode(top, message, scale);
        heongpu::Ciphertext<scheme> dropped(context);
        encryptor.encrypt(dropped, top);
        for (int i = 0; i < depth; i++)
        {
            operators->mod_drop_inplace(dropped);
        }
        operators->add_inplace(ciphertext, dropped);

        heongpu::Plaintext<scheme> result_plaintext(context);
        decryptor.decrypt(result_plaintext, ciphertext);
        std::vector<double> result;
        encoder->decode(result, result_plaintext);
        for (int i = 0; i < slots; i++)
        {
            ASSERT_NEAR(result[i], 2.0 * message[i], 1e-3) << "slot " << i;
        }
    }
}

TEST_F(CKKSEncodeDepth, Invalid_Depth_Is_Rejected)
{
    heongpu::Plaintext<scheme> plaintext(context);
    EXPECT_THROW(encoder->encode(plaintext, message, scale, -1),
                 heongpu::ParameterError);
    EXPECT_THROW(encoder->encode(plaintext, message, scale, 3),
                 heongpu::ParameterError);
}
