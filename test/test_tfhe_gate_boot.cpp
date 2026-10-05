// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <random>

constexpr auto Scheme = heongpu::Scheme::TFHE;

TEST(HEonGPU, TFHE_Gate_Boots)
{
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Bootstrappingkey<Scheme> boot_key(context);
    keygen.generate_bootstrapping_key(boot_key, secret_key);

    heongpu::HEEncryptor<Scheme> encryptor(context, secret_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HELogicOperator<Scheme> logic(context);

    constexpr size_t size = 64;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1);

    std::vector<bool> input1(size), input2(size), control(size);
    for (size_t i = 0; i < size; ++i)
    {
        input1[i] = dis(gen);
        input2[i] = dis(gen);
        control[i] = dis(gen);
    }

    std::vector<bool> expected_nand(size), expected_and(size),
        expected_nor(size), expected_or(size);
    std::vector<bool> expected_xnor(size), expected_xor(size),
        expected_not(size), expected_mux(size);

    for (size_t i = 0; i < size; ++i)
    {
        expected_nand[i] = !(input1[i] & input2[i]);
        expected_and[i] = input1[i] & input2[i];
        expected_nor[i] = !(input1[i] | input2[i]);
        expected_or[i] = input1[i] | input2[i];
        expected_xnor[i] = !(input1[i] ^ input2[i]);
        expected_xor[i] = input1[i] ^ input2[i];
        expected_not[i] = !input1[i];
        expected_mux[i] = control[i] ? input1[i] : input2[i];
    }

    heongpu::Ciphertext<Scheme> ct1(context);
    heongpu::Ciphertext<Scheme> ct2(context);
    heongpu::Ciphertext<Scheme> ct3(context);
    encryptor.encrypt(ct1, input1);
    encryptor.encrypt(ct2, input2);
    encryptor.encrypt(ct3, control);

    auto check_gate = [&](auto&& gate_func, const auto& expected)
    {
        heongpu::Ciphertext<Scheme> result(context);
        gate_func(result);
        std::vector<bool> decrypted;
        decryptor.decrypt(result, decrypted);
        EXPECT_EQ(decrypted, expected);
    };

    check_gate([&](auto& r) { logic.NAND(ct1, ct2, r, boot_key); },
               expected_nand);
    check_gate([&](auto& r) { logic.AND(ct1, ct2, r, boot_key); },
               expected_and);
    check_gate([&](auto& r) { logic.NOR(ct1, ct2, r, boot_key); },
               expected_nor);
    check_gate([&](auto& r) { logic.OR(ct1, ct2, r, boot_key); }, expected_or);
    check_gate([&](auto& r) { logic.XNOR(ct1, ct2, r, boot_key); },
               expected_xnor);
    check_gate([&](auto& r) { logic.XOR(ct1, ct2, r, boot_key); },
               expected_xor);
    check_gate([&](auto& r) { logic.NOT(ct1, r); }, expected_not);
    check_gate([&](auto& r) { logic.MUX(ct1, ct2, ct3, r, boot_key); },
               expected_mux);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
