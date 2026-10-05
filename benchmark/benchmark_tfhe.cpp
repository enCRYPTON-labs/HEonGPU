// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <random>
#include <cuda_runtime.h>
#include <iostream>

constexpr auto Scheme = heongpu::Scheme::TFHE;

int main()
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

    constexpr size_t size = 8;
    constexpr int repeat_count = 50;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto benchmark_gate = [&](const std::string& name, auto&& gate_func,
                              const std::vector<bool>& input1,
                              const std::vector<bool>& input2,
                              const std::vector<bool>& control)
    {
        float accumulated_time = 0;

        for (int r = 0; r < repeat_count; ++r)
        {
            heongpu::Ciphertext<Scheme> ct1(context);
            heongpu::Ciphertext<Scheme> ct2(context);
            heongpu::Ciphertext<Scheme> ct3(context);

            encryptor.encrypt(ct1, input1);
            encryptor.encrypt(ct2, input2);
            encryptor.encrypt(ct3, control);

            heongpu::Ciphertext<Scheme> result(context);
            cudaEventRecord(start);
            gate_func(ct1, ct2, ct3, result);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            accumulated_time += ms;
        }

        std::cout << "[" << name
                  << "] Avg Time: " << (accumulated_time / repeat_count)
                  << " ms" << std::endl;
    };

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

    benchmark_gate(
        "NAND",
        [&](auto& a, auto& b, auto&, auto& r)
        { logic.NAND(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "AND",
        [&](auto& a, auto& b, auto&, auto& r) { logic.AND(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "NOR",
        [&](auto& a, auto& b, auto&, auto& r) { logic.NOR(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "OR",
        [&](auto& a, auto& b, auto&, auto& r) { logic.OR(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "XNOR",
        [&](auto& a, auto& b, auto&, auto& r)
        { logic.XNOR(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "XOR",
        [&](auto& a, auto& b, auto&, auto& r) { logic.XOR(a, b, r, boot_key); },
        input1, input2, control);
    benchmark_gate(
        "NOT", [&](auto& a, auto&, auto&, auto& r) { logic.NOT(a, r); }, input1,
        input2, control);
    std::cout << "The MUX will be even faster soon! -> ";
    benchmark_gate(
        "MUX",
        [&](auto& a, auto& b, auto& c, auto& r)
        { logic.MUX(a, b, c, r, boot_key); },
        input1, input2, control);

    return 0;
}
