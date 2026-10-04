// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

int main(int argc, char* argv[])
{
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::sec128);
    size_t poly_modulus_degree = 1 << 16;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Q0 | 3 SlotToCoeff + 12 user levels (9 new multiplicative depth) | 8
    // EvalMod levels | 4 CoeffToSlot
    context->set_coeff_modulus_bit_sizes({60, 40, 40, 40, 40, 40, 40, 40, 40,
                                          40, 40, 40, 40, 60, 60, 60, 60, 60,
                                          60, 60, 60, 56, 56, 56, 56},
                                         {60, 60, 60, 60, 60});
    context->generate();
    context->print_parameters();

    int h = 192;
    int ephemeral_secret_weight = 32;

    double scale = pow(2.0, 40);

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, h);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::Secretkey<heongpu::Scheme::CKKS> sparse_secret_key(
        context, ephemeral_secret_weight);
    keygen.generate_secret_key(sparse_secret_key);

    heongpu::Switchkey<heongpu::Scheme::CKKS> swk_dense_to_sparse(context);
    keygen.generate_switch_key(swk_dense_to_sparse, sparse_secret_key,
                               secret_key);
    heongpu::Switchkey<heongpu::Scheme::CKKS> swk_sparse_to_dense(context);
    keygen.generate_switch_key(swk_sparse_to_dense, secret_key,
                               sparse_secret_key);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                   encoder);

    const int slot_count = poly_modulus_degree / 2;
    std::vector<double>
        message; // Message has to be Real Number for Slim bootstrapping
    for (int i = 0; i < slot_count; i++)
    {
        message.push_back(0.2);
    }

    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, message, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    // Slim bootstrapping starts with SlotToCoeff, so slot values become
    // coefficients directly; a larger message ratio keeps sin(x) ~ x accurate.
    heongpu::EvalModConfig eval_mod_config(20);
    eval_mod_config.message_ratio_ = 1024.0;

    heongpu::BootstrappingConfig boot_config(
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::SLOTS_TO_COEFFS, 3),
        eval_mod_config,
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::COEFFS_TO_SLOTS, 24));

    operators.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);

    std::vector<int> key_index = operators.bootstrapping_key_indexs();
    std::cout << "Total galois key needed for CKKS bootstrapping: "
              << key_index.size() << std::endl;
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret_key);

    // Drop levels until the SlotToCoeff start level
    while (C1.level() > 3)
    {
        operators.mod_drop_inplace(C1);
    }

    std::cout << "Level before bootstrapping: " << C1.level() << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operators.slim_bootstrapping(C1, galois_key, relin_key,
                                     &swk_dense_to_sparse,
                                     &swk_sparse_to_dense);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Level after bootstrapping: " << cipher_boot.level()
              << std::endl;
    std::cout << "Bootstrapping time: " << milliseconds << " ms ("
              << milliseconds / 1000.0 << " seconds)" << std::endl;

    heongpu::Plaintext<heongpu::Scheme::CKKS> P_res1(context);
    decryptor.decrypt(P_res1, cipher_boot);
    std::vector<Complex64> decrypted_1;
    encoder.decode(decrypted_1, P_res1);

    heongpu::PrecisionStats prec_stats =
        heongpu::get_precision_stats(message, decrypted_1);

    std::cout << "\n=== Bootstrapping Precision Statistics ===" << std::endl;
    std::cout << prec_stats.to_string() << std::endl;

    for (int j = 0; j < 16; j++)
    {
        std::cout << j << "-> EXPECTED:" << message[j]
                  << " - ACTUAL:" << decrypted_1[j] << std::endl;
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
