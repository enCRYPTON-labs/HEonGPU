// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <iostream>
#include <vector>

// Use TFHE scheme
constexpr auto Scheme = heongpu::Scheme::TFHE;

int main(int argc, char* argv[])
{
    // ========================================
    // 1. Initialize TFHE encryption context
    // ========================================
    // Currently, a fixed parameter set is used for the TFHE scheme,
    // providing 128-bit security. In future releases, configurable
    // and lattice-estimator-based parameter sets targeting 128, 192,
    // and 256-bit security levels will also be supported.
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // ========================================
    // 2. Key Generation
    // ========================================
    heongpu::HEKeyGenerator<Scheme> keygen(context);

    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Bootstrappingkey<Scheme> boot_key(context);
    keygen.generate_bootstrapping_key(boot_key, secret_key);

    // ========================================
    // 3. Setup Encryptor, Decryptor, and Logic Operator
    // ========================================
    heongpu::HEEncryptor<Scheme> encryptor(context, secret_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HELogicOperator<Scheme> logic(context);

    // ========================================
    // 4. Define Input Messages (plaintext)
    // ========================================
    std::vector<bool> input1 = {1, 1, 0, 1, 0, 1, 0, 0};
    std::vector<bool> input2 = {1, 0, 1, 0, 1, 1, 1, 0};
    std::vector<bool> input3 = {0, 0, 0, 0, 1, 1, 1, 1};

    std::cout << "Input1: ";
    for (auto b : input1)
        std::cout << b << ", ";
    std::cout << std::endl;

    std::cout << "Input2: ";
    for (auto b : input2)
        std::cout << b << ", ";
    std::cout << std::endl;

    std::cout << "Input3(control input of MUX): ";
    for (auto b : input3)
        std::cout << b << ", ";
    std::cout << std::endl;

    // ========================================
    // 5. Encrypt input vectors
    // ========================================
    heongpu::Ciphertext<Scheme> ct1(context);
    heongpu::Ciphertext<Scheme> ct2(context);
    heongpu::Ciphertext<Scheme> ct3(context);
    encryptor.encrypt(ct1, input1);
    encryptor.encrypt(ct2, input2);
    encryptor.encrypt(ct3, input3);

    // ========================================
    // 6. Apply Logic Gates and Decrypt Results
    // ========================================

    // --- NAND ---
    heongpu::Ciphertext<Scheme> nand_result(context);
    logic.NAND(ct1, ct2, nand_result, boot_key);

    std::vector<bool> nand_decrypted;
    decryptor.decrypt(nand_result, nand_decrypted);

    std::cout << "NAND (Decrypted):  ";
    for (bool b : nand_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [0, 1, 1, 1, 1, 0, 1, 1]

    // --- AND ---
    heongpu::Ciphertext<Scheme> and_result(context);
    logic.AND(ct1, ct2, and_result, boot_key);

    std::vector<bool> and_decrypted;
    decryptor.decrypt(and_result, and_decrypted);

    std::cout << "AND (Decrypted):   ";
    for (bool b : and_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [1, 0, 0, 0, 0, 1, 0, 0]

    // --- NOR ---
    heongpu::Ciphertext<Scheme> nor_result(context);
    logic.NOR(ct1, ct2, nor_result, boot_key);

    std::vector<bool> nor_decrypted;
    decryptor.decrypt(nor_result, nor_decrypted);

    std::cout << "NOR (Decrypted):   ";
    for (bool b : nor_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [0, 0, 0, 0, 0, 0, 0, 1]

    // --- OR ---
    heongpu::Ciphertext<Scheme> or_result(context);
    logic.OR(ct1, ct2, or_result, boot_key);

    std::vector<bool> or_decrypted;
    decryptor.decrypt(or_result, or_decrypted);

    std::cout << "OR (Decrypted):    ";
    for (bool b : or_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [1, 1, 1, 1, 1, 1, 1, 0]

    // --- XNOR ---
    heongpu::Ciphertext<Scheme> xnor_result(context);
    logic.XNOR(ct1, ct2, xnor_result, boot_key);

    std::vector<bool> xnor_decrypted;
    decryptor.decrypt(xnor_result, xnor_decrypted);

    std::cout << "XNOR (Decrypted):  ";
    for (bool b : xnor_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [1, 0, 0, 0, 0, 1, 0, 1]

    // --- XOR ---
    heongpu::Ciphertext<Scheme> xor_result(context);
    logic.XOR(ct1, ct2, xor_result, boot_key);

    std::vector<bool> xor_decrypted;
    decryptor.decrypt(xor_result, xor_decrypted);

    std::cout << "XOR (Decrypted):   ";
    for (bool b : xor_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [0, 1, 1, 1, 1, 0, 1, 0]

    // --- NOT ---
    heongpu::Ciphertext<Scheme> not_result(context);
    logic.NOT(ct1, not_result); // NOT gate does not require bootstrapping key

    std::vector<bool> not_decrypted;
    decryptor.decrypt(not_result, not_decrypted);

    std::cout << "NOT input1 (Decrypted): ";
    for (bool b : not_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [0, 0, 1, 0, 1, 0, 1, 1]

    // --- MUX ---
    heongpu::Ciphertext<Scheme> mux_result(context);
    logic.MUX(ct1, ct2, ct3, mux_result, boot_key);

    std::vector<bool> mux_decrypted;
    decryptor.decrypt(mux_result, mux_decrypted);

    std::cout << "MUX (Decrypted): ";
    for (bool b : mux_decrypted)
        std::cout << b << ", ";
    std::cout << std::endl;
    // (Expected): [1, 0, 1, 0, 0, 1, 0, 0]

    return EXIT_SUCCESS;
}
