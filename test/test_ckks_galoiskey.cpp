// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

// Rotations by a shift without its own Galois key are built from the power of
// two keys. These tests check that this keeps working for keys that are
// copied, moved, loaded or built from a custom list of shifts, and that a
// rotation that cannot be built is reported instead of giving a wrong result.

namespace
{
    using heongpu::CudaException;
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;
    constexpr int slots = degree / 2;

    class CKKSGaloiskey : public ::testing::TestWithParam<std::vector<int>>
    {
      protected:
        void SetUp() override
        {
            // One P prime selects KEYSWITCHING_METHOD_I, two select
            // KEYSWITCHING_METHOD_II.
            context =
                heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes({40, 30, 30}, GetParam());
            context->generate();

            generator =
                std::make_unique<heongpu::HEKeyGenerator<scheme>>(context);
            secret_key = heongpu::Secretkey<scheme>(context);
            generator->generate_secret_key(secret_key);
            public_key = heongpu::Publickey<scheme>(context);
            generator->generate_public_key(public_key, secret_key);
            encoder = std::make_unique<heongpu::HEEncoder<scheme>>(context);
            operators = std::make_unique<heongpu::HEArithmeticOperator<scheme>>(
                context, *encoder);

            message.resize(slots);
            for (int i = 0; i < slots; i++)
            {
                message[i] = static_cast<double>(i % 97) / 97.0;
            }
        }

        heongpu::Ciphertext<scheme> encrypt()
        {
            heongpu::HEEncryptor<scheme> encryptor(context, public_key);
            heongpu::Plaintext<scheme> plaintext(context);
            encoder->encode(plaintext, message, pow(2.0, 30));
            heongpu::Ciphertext<scheme> ciphertext(context);
            encryptor.encrypt(ciphertext, plaintext);
            return ciphertext;
        }

        // Rotates a fresh ciphertext into a separate output and checks the
        // decrypted result.
        void expect_rotation(heongpu::Galoiskey<scheme>& galois_key, int shift)
        {
            auto input = encrypt();
            heongpu::Ciphertext<scheme> output(context);
            operators->rotate_rows(input, output, galois_key, shift);

            heongpu::HEDecryptor<scheme> decryptor(context, secret_key);
            heongpu::Plaintext<scheme> plaintext(context);
            decryptor.decrypt(plaintext, output);
            std::vector<double> result;
            encoder->decode(result, plaintext);
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

            ASSERT_EQ(result.size(), message.size());
            double max_error = 0;
            for (int i = 0; i < slots; i++)
            {
                const int index = ((i + shift) % slots + slots) % slots;
                max_error =
                    std::max(max_error, std::fabs(result[i] - message[index]));
            }
            EXPECT_LT(max_error, 1e-2) << "shift " << shift;
        }

        heongpu::Galoiskey<scheme> default_key()
        {
            heongpu::Galoiskey<scheme> galois_key(context);
            generator->generate_galois_key(galois_key, secret_key);
            return galois_key;
        }

        heongpu::HEContext<scheme> context;
        std::unique_ptr<heongpu::HEKeyGenerator<scheme>> generator;
        heongpu::Secretkey<scheme> secret_key;
        heongpu::Publickey<scheme> public_key;
        std::unique_ptr<heongpu::HEEncoder<scheme>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<scheme>> operators;
        std::vector<double> message;
    };
} // namespace

TEST_P(CKKSGaloiskey, Default_Key_Builds_Other_Shifts)
{
    auto galois_key = default_key();
    expect_rotation(galois_key, 3);
    expect_rotation(galois_key, -6);
}

TEST_P(CKKSGaloiskey, Copied_Key_Builds_Other_Shifts)
{
    auto original = default_key();
    heongpu::Galoiskey<scheme> copy(original);
    expect_rotation(copy, 3);

    heongpu::Galoiskey<scheme> assigned;
    assigned = original;
    expect_rotation(assigned, -6);
}

TEST_P(CKKSGaloiskey, Moved_Key_Builds_Other_Shifts)
{
    auto original = default_key();
    heongpu::Galoiskey<scheme> moved(std::move(original));
    expect_rotation(moved, 3);

    heongpu::Galoiskey<scheme> assigned;
    assigned = std::move(moved);
    expect_rotation(assigned, -6);
}

TEST_P(CKKSGaloiskey, Loaded_Key_Builds_Other_Shifts)
{
    auto original = default_key();
    std::stringstream stream;
    original.save(stream);
    heongpu::Galoiskey<scheme> loaded;
    loaded.load(stream);
    expect_rotation(loaded, 3);
    expect_rotation(loaded, -6);
}

TEST_P(CKKSGaloiskey, Custom_Shifts_Build_Other_Shifts)
{
    std::vector<int> shifts = {1, 2};
    heongpu::Galoiskey<scheme> galois_key(context, shifts);
    generator->generate_galois_key(galois_key, secret_key);
    expect_rotation(galois_key, 1);
    expect_rotation(galois_key, 3);
    expect_rotation(galois_key, 7);
}

TEST_P(CKKSGaloiskey, Missing_Shift_Is_Reported)
{
    // No power of two keys: only the listed shift can be computed.
    std::vector<int> shifts = {3};
    heongpu::Galoiskey<scheme> galois_key(context, shifts);
    generator->generate_galois_key(galois_key, secret_key);
    expect_rotation(galois_key, 3);

    auto input = encrypt();
    heongpu::Ciphertext<scheme> output(context);
    EXPECT_THROW(operators->rotate_rows(input, output, galois_key, 5),
                 heongpu::KeyNotFoundError);

    // Negative steps need the negative power of two keys.
    std::vector<int> positive = {1};
    heongpu::Galoiskey<scheme> positive_key(context, positive);
    generator->generate_galois_key(positive_key, secret_key);
    EXPECT_THROW(operators->rotate_rows(input, output, positive_key, -1),
                 heongpu::KeyNotFoundError);
}

TEST_P(CKKSGaloiskey, Full_Rotation_Returns_The_Input)
{
    auto galois_key = default_key();
    expect_rotation(galois_key, slots);
    expect_rotation(galois_key, -slots);
}

TEST_P(CKKSGaloiskey, Shifts_Beyond_The_Slot_Count_Wrap_Around)
{
    auto galois_key = default_key();
    expect_rotation(galois_key, slots + 3);
    expect_rotation(galois_key, -slots - 6);
}

TEST_P(CKKSGaloiskey, Key_For_A_Shift_Beyond_The_Slot_Count_Is_Rejected)
{
    std::vector<int> shifts = {slots + 1};
    EXPECT_THROW(heongpu::Galoiskey<scheme>(context, shifts),
                 heongpu::ParameterError);
}

INSTANTIATE_TEST_SUITE_P(HEonGPU, CKKSGaloiskey,
                         ::testing::Values(std::vector<int>{40},
                                           std::vector<int>{40, 40}));
