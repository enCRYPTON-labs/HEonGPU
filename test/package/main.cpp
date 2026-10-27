// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Encrypts, adds and decrypts with an installed HEonGPU. Needs a GPU to run;
// building and linking it already checks the installed package.

#include <heongpu/heongpu.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    constexpr auto scheme = heongpu::Scheme::CKKS;
    auto context = heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();
    std::cout << context << std::endl;

    heongpu::HEKeyGenerator<scheme> generator(context);
    heongpu::Secretkey<scheme> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<scheme> public_key(context);
    generator.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<scheme> encoder(context);
    heongpu::HEEncryptor<scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<scheme> operators(context, encoder);

    heongpu::Plaintext<scheme> plaintext(context);
    encoder.encode(plaintext, std::vector<double>(2048, 0.5), pow(2.0, 30));
    heongpu::Ciphertext<scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);
    operators.add_inplace(ciphertext, ciphertext);
    decryptor.decrypt(plaintext, ciphertext);

    std::vector<double> result;
    encoder.decode(result, plaintext);
    std::cout << "0.5 + 0.5 = " << result[0] << std::endl;
    return std::fabs(result[0] - 1.0) < 1e-3 ? 0 : 1;
}
