// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <heongpu/heongpu.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // Create CKKS context for coefficient/slot transform demonstration.
    heongpu::HEContext<Scheme> context =
        heongpu::GenHEContext<Scheme>(heongpu::sec_level_type::none);

    const std::size_t poly_modulus_degree = 4096;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({50, 40, 40, 40, 40, 40, 40, 40, 40},
                                         {50, 50});
    context->generate();

    const double scale = std::pow(2.0, 40);

    // Generate keys and CKKS helpers.
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // Build transform context and Vandermonde-related precomputations.
    // ctos_start_level and stoc_start_level define where transforms begin.
    const int ctos_start_level = 0;
    const int stoc_start_level = 3;
    heongpu::CKKSEncodingTransformConfig transform_config(
        3, 3, ctos_start_level, stoc_start_level);
    heongpu::CKKSEncodingTransformContext transform_context;
    operators.generate_encoding_transform_context(transform_context,
                                                  transform_config);

    std::vector<int> key_index = transform_context.key_indexs_;
    heongpu::Galoiskey<Scheme> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret_key);

    // Prepare coefficient-encoded message and encrypt it.
    std::vector<double> coeff_message(poly_modulus_degree, 0.0);
    coeff_message[0] = 1.00;
    coeff_message[1] = -0.50;
    coeff_message[2] = 0.25;
    coeff_message[3] = -1.75;
    coeff_message[4] = 2.50;
    coeff_message[5] = -3.25;
    coeff_message[6] = 0.125;
    coeff_message[7] = -0.875;

    heongpu::Plaintext<Scheme> coeff_plain(context);
    encoder.encode(coeff_plain, coeff_message, scale, heongpu::ExecutionOptions(),
                   heongpu::encoding::COEFFICIENT);

    heongpu::Ciphertext<Scheme> coeff_cipher(context);
    encryptor.encrypt(coeff_cipher, coeff_plain);

    // Convert coefficient-domain ciphertext into two slot-domain ciphertexts.
    std::vector<heongpu::Ciphertext<Scheme>> slot_pair =
        operators.coeff_to_slot(coeff_cipher, galois_key, transform_context);

    if ((slot_pair[0].encoding_type() != heongpu::encoding::SLOT) ||
        (slot_pair[1].encoding_type() != heongpu::encoding::SLOT))
    {
        std::cerr << "coeff_to_slot output is not slot-encoded." << std::endl;
        return EXIT_FAILURE;
    }

    // Convert back to coefficient encoding (roundtrip).
    heongpu::Ciphertext<Scheme> coeff_roundtrip =
        operators.slot_to_coeff(slot_pair[0], slot_pair[1], galois_key,
                                transform_context);

    heongpu::Plaintext<Scheme> ref_plain(context);
    decryptor.decrypt(ref_plain, coeff_cipher);
    std::vector<double> ref_coeff;
    encoder.decode(ref_coeff, ref_plain);

    heongpu::Plaintext<Scheme> roundtrip_plain(context);
    decryptor.decrypt(roundtrip_plain, coeff_roundtrip);
    std::vector<double> roundtrip_coeff;
    encoder.decode(roundtrip_coeff, roundtrip_plain);

    // Compare decrypted reference and roundtrip output.
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < ref_coeff.size(); ++i)
    {
        max_abs_error = std::max(
            max_abs_error, std::abs(ref_coeff[i] - roundtrip_coeff[i]));
    }

    std::cout << "Input ciphertext encoding      : "
              << static_cast<int>(coeff_cipher.encoding_type()) << std::endl;
    std::cout << "coeff_to_slot output encoding0 : "
              << static_cast<int>(slot_pair[0].encoding_type()) << std::endl;
    std::cout << "coeff_to_slot output encoding1 : "
              << static_cast<int>(slot_pair[1].encoding_type()) << std::endl;
    std::cout << "slot_to_coeff output encoding  : "
              << static_cast<int>(coeff_roundtrip.encoding_type()) << std::endl;
    std::cout << "Max abs error (roundtrip)      : " << max_abs_error
              << std::endl;

    if (coeff_roundtrip.encoding_type() != heongpu::encoding::COEFFICIENT)
    {
        std::cerr << "slot_to_coeff output is not coefficient-encoded."
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (max_abs_error > 5e-2)
    {
        std::cerr << "coeff_to_slot/slot_to_coeff roundtrip test failed."
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "CKKS coeff_to_slot/slot_to_coeff test passed." << std::endl;
    return EXIT_SUCCESS;
}
