// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <heongpu/heongpu.hpp>
#include <iostream>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // Create CKKS context with user-defined parameters.
    heongpu::HEParams<Scheme> params;
    params.security_level = heongpu::sec_level_type::none;
    const std::size_t poly_modulus_degree = 8192;
    params.poly_modulus_degree = poly_modulus_degree;
    params.coeff_modulus_bit_sizes = {60, 40, 40, 60};
    params.special_modulus_bit_sizes = {60};
    heongpu::HEContext<Scheme> context = params.make_context();
    context->generate();

    const double scale = std::pow(2.0, 40);

    // Generate basic keys and CKKS helpers.
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // In coefficient encoding, values are interpreted as polynomial
    // coefficients.
    std::vector<double> coeff_message(poly_modulus_degree, 0.0);
    coeff_message[0] = 1.25;
    coeff_message[1] = -2.0;
    coeff_message[2] = 3.5;
    coeff_message[3] = -0.125;
    coeff_message[4] = 0.75;
    coeff_message[5] = -1.5;

    heongpu::Plaintext<Scheme> coeff_plain(context);
    encoder.encode(coeff_plain, coeff_message, scale,
                   heongpu::ExecutionOptions(), heongpu::encoding::coefficient);

    // Encrypt -> decrypt -> decode and compare with the original coefficients.
    heongpu::Ciphertext<Scheme> coeff_cipher(context);
    encryptor.encrypt(coeff_cipher, coeff_plain);

    heongpu::Plaintext<Scheme> coeff_plain_dec(context);
    decryptor.decrypt(coeff_plain_dec, coeff_cipher);

    std::vector<double> coeff_result;
    encoder.decode(coeff_result, coeff_plain_dec);

    // Compute maximum absolute reconstruction error.
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < coeff_message.size(); ++i)
    {
        max_abs_error = std::max(max_abs_error,
                                 std::abs(coeff_message[i] - coeff_result[i]));
    }

    std::cout << "Plaintext encoding  : "
              << static_cast<int>(coeff_plain.encoding_type()) << std::endl;
    std::cout << "Ciphertext encoding : "
              << static_cast<int>(coeff_cipher.encoding_type()) << std::endl;
    std::cout << "Decoded encoding    : "
              << static_cast<int>(coeff_plain_dec.encoding_type()) << std::endl;
    std::cout << "Max abs error       : " << max_abs_error << std::endl;

    std::cout << "First 6 coefficients (expected -> decoded):" << std::endl;
    for (int i = 0; i < 6; ++i)
    {
        std::cout << "  [" << i << "] " << coeff_message[i] << " -> "
                  << coeff_result[i] << std::endl;
    }

    return EXIT_SUCCESS;
}
