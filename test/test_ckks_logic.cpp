// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

// Every gate of the CKKS logic operator, on two ciphertexts and on a
// ciphertext and a plaintext, gives its truth table on the four combinations
// of input bits.

namespace
{
    constexpr heongpu::Scheme S = heongpu::Scheme::CKKS;
    using Cipher = heongpu::Ciphertext<S>;
    using Plain = heongpu::Plaintext<S>;

    class CkksLogic : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            heongpu::HEParams<S> params;
            params.security_level = heongpu::sec_level_type::none;
            params.poly_modulus_degree = 8192;
            params.coeff_modulus_bit_sizes = {60, 40, 40, 40};
            params.special_modulus_bit_sizes = {60};
            context = params.make_context();
            context->generate();
            slot_count = params.poly_modulus_degree / 2;

            heongpu::HEKeyGenerator<S> keygen(context);
            secret_key = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen.generate_secret_key(*secret_key);
            public_key = std::make_unique<heongpu::Publickey<S>>(context);
            keygen.generate_public_key(*public_key, *secret_key);
            relin_key = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen.generate_relin_key(*relin_key, *secret_key);

            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret_key);
            operators = std::make_unique<heongpu::HELogicOperator<S>>(
                context, *encoder, scale);

            for (int i = 0; i < slot_count; i++)
            {
                a.push_back(i % 2);
                b.push_back((i / 2) % 2);
            }
        }

        Plain encode(const std::vector<double>& values)
        {
            Plain plaintext(context);
            encoder->encode(plaintext, values, scale);
            return plaintext;
        }

        Cipher encrypt(const std::vector<double>& values)
        {
            Plain plaintext = encode(values);
            Cipher ciphertext(context);
            encryptor->encrypt(ciphertext, plaintext);
            return ciphertext;
        }

        void check(const std::string& gate, Cipher& ciphertext,
                   const std::function<int(int, int)>& expected)
        {
            Plain plaintext(context);
            decryptor->decrypt(plaintext, ciphertext);
            std::vector<double> result;
            encoder->decode(result, plaintext);
            for (int i = 0; i < slot_count; i++)
            {
                ASSERT_NEAR(result[i], expected(a[i], b[i]), 1e-3)
                    << gate << ", slot " << i;
            }
        }

        heongpu::HEContext<S> context;
        double scale = std::pow(2.0, 40);
        int slot_count = 0;
        std::unique_ptr<heongpu::Secretkey<S>> secret_key;
        std::unique_ptr<heongpu::Publickey<S>> public_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin_key;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HELogicOperator<S>> operators;
        std::vector<double> a, b;
    };

    int and_(int x, int y)
    {
        return x & y;
    }
    int or_(int x, int y)
    {
        return x | y;
    }
    int xor_(int x, int y)
    {
        return x ^ y;
    }
    int nand_(int x, int y)
    {
        return 1 - (x & y);
    }
    int nor_(int x, int y)
    {
        return 1 - (x | y);
    }
    int xnor_(int x, int y)
    {
        return 1 - (x ^ y);
    }
} // namespace

TEST_F(CkksLogic, GatesOnTwoCiphertexts)
{
    Cipher x = encrypt(a);
    Cipher y = encrypt(b);
    auto& op = *operators;
    auto& rk = *relin_key;

    Cipher output(context);
    op.AND(x, y, output, rk);
    check("AND", output, and_);
    op.OR(x, y, output, rk);
    check("OR", output, or_);
    op.XOR(x, y, output, rk);
    check("XOR", output, xor_);
    op.NAND(x, y, output, rk);
    check("NAND", output, nand_);
    op.NOR(x, y, output, rk);
    check("NOR", output, nor_);
    op.XNOR(x, y, output, rk);
    check("XNOR", output, xnor_);
    op.NOT(x, output);
    check("NOT", output, [](int x, int) { return 1 - x; });
}

TEST_F(CkksLogic, GatesOnACiphertextAndAPlaintext)
{
    Cipher x = encrypt(a);
    Plain y = encode(b);
    auto& op = *operators;

    Cipher output(context);
    op.AND(x, y, output);
    check("AND", output, and_);
    op.OR(x, y, output);
    check("OR", output, or_);
    op.XOR(x, y, output);
    check("XOR", output, xor_);
    op.NAND(x, y, output);
    check("NAND", output, nand_);
    op.NOR(x, y, output);
    check("NOR", output, nor_);
    op.XNOR(x, y, output);
    check("XNOR", output, xnor_);
}

TEST_F(CkksLogic, InPlaceGates)
{
    Cipher y = encrypt(b);
    Plain p = encode(b);
    auto& op = *operators;
    auto& rk = *relin_key;

    Cipher x = encrypt(a);
    op.OR_inplace(x, y, rk);
    check("OR_inplace", x, or_);
    x = encrypt(a);
    op.XOR_inplace(x, y, rk);
    check("XOR_inplace", x, xor_);
    x = encrypt(a);
    op.NOR_inplace(x, p);
    check("NOR_inplace", x, nor_);
    x = encrypt(a);
    op.XNOR_inplace(x, p);
    check("XNOR_inplace", x, xnor_);
}
