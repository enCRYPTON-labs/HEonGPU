// Copyright 2025-2026 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

int main(int argc, char* argv[])
{
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::sec128);
    size_t poly_modulus_degree = 1 << 16;
    context->set_poly_modulus_degree(poly_modulus_degree);

    context->set_coeff_modulus_values(
        {
            0x10000000006e0001, // 60 Q0
            0x10000140001, // 40
            0xffffe80001, // 40
            0xffffc40001, // 40
            0x100003e0001, // 40
            0xffffb20001, // 40
            0x10000500001, // 40
            0xffff940001, // 40
            0xffff8a0001, // 40
            0xffff820001, // 40
            0x7fffe60001, // 39 StC
            0x7fffe40001, // 39 StC
            0x7fffe00001, // 39 StC
            0xfffffffff840001, // 60 Sine (double angle)
            0x1000000000860001, // 60 Sine (double angle)
            0xfffffffff6a0001, // 60 Sine (double angle)
            0x1000000000980001, // 60 Sine
            0xfffffffff5a0001, // 60 Sine
            0x1000000000b00001, // 60 Sine
            0x1000000000ce0001, // 60 Sine
            0xfffffffff2a0001, // 60 Sine
            0x100000000060001, // 56 CtS
            0xfffffffff00001, // 56 CtS
            0xffffffffd80001, // 56 CtS
            0x1000000002a0001, // 56 CtS
        },
        {
            0x1fffffffffe00001, // Pi 61
            0x1fffffffffc80001, // Pi 61
            0x1fffffffffb40001, // Pi 61
            0x1fffffffff500001, // Pi 61
            0x1fffffffff420001, // Pi 61
        });

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

    heongpu::Switchkey<heongpu::Scheme::CKKS>* swk_dense_to_sparse = nullptr;
    heongpu::Switchkey<heongpu::Scheme::CKKS>* swk_sparse_to_dense = nullptr;

    if (ephemeral_secret_weight > 0)
    {
        heongpu::Secretkey<heongpu::Scheme::CKKS> sparse_secret_key(
            context, ephemeral_secret_weight);
        keygen.generate_secret_key(sparse_secret_key);

        swk_dense_to_sparse =
            new heongpu::Switchkey<heongpu::Scheme::CKKS>(context);
        keygen.generate_switch_key(*swk_dense_to_sparse, sparse_secret_key,
                                   secret_key); // dense->sparse

        swk_sparse_to_dense =
            new heongpu::Switchkey<heongpu::Scheme::CKKS>(context);
        keygen.generate_switch_key(*swk_sparse_to_dense, secret_key,
                                   sparse_secret_key); // sparse->dense
    }

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                   encoder);

    const int slot_count = poly_modulus_degree / 2;
    std::vector<Complex64> message;
    for (int i = 0; i < slot_count; i++)
    {
        message.push_back(Complex64(0.2, 0.4));
    }

    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, message, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    heongpu::EvalModConfig eval_mod_config(20);

    heongpu::BootstrappingConfig boot_config(
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::SLOTS_TO_COEFFS, 12),
        eval_mod_config,
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::COEFFS_TO_SLOTS, 24));

    operators.generate_bootstrapping_params(scale, boot_config);

    std::vector<int> key_index = operators.bootstrapping_key_indices();
    std::cout << "Total galois key needed for CKKS bootstrapping: "
              << key_index.size() << std::endl;
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret_key);

    // Drop all level until one level remain
    for (int i = 0; i < 24; i++)
    {
        operators.mod_drop_inplace(C1);
    }

    std::cout << "Level before bootstrapping: " << C1.level() << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operators.regular_bootstrapping(C1, galois_key, relin_key,
                                        swk_dense_to_sparse,
                                        swk_sparse_to_dense);
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

    // Compute and print precision statistics
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
