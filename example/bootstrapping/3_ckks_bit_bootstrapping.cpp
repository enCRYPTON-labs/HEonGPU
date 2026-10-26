// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

int main(int argc, char* argv[])
{
    // N = 2^16, the moduli, the scale, the Hamming weights of the secret keys
    // and the bootstrapping configuration of this example.
    heongpu::CKKSBootstrappingPreset preset =
        heongpu::CKKSBootstrappingPreset::bit();

    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::sec128);
    preset.apply(context);
    context->generate();
    context->print_parameters();

    double scale = preset.scale;

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(
        context, preset.secret_key_hamming_weight);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    heongpu::HELogicOperator<heongpu::Scheme::CKKS> operators(context, encoder,
                                                              scale);

    operators.generate_bootstrapping_params(preset);

    // The relinearization key, the Galois keys of the bootstrapping and the
    // switch keys between the secret key and a sparse secret key.
    std::vector<int> key_index = operators.bootstrapping_key_indices();
    std::cout << "Total galois key needed for CKKS bootstrapping: "
              << key_index.size() << std::endl;
    heongpu::CKKSBootstrappingKeys boot_keys = keygen.bootstrapping_keys(
        secret_key, key_index, preset.sparse_secret_key_hamming_weight);

    const int slot_count = preset.poly_modulus_degree / 2;
    std::vector<double> message(slot_count, 1);
    message[0] = 0;
    message[1] = 0;

    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, message, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    // Bootstrapping brings its input down to the level it starts from.
    std::cout << "Level before bootstrapping: " << C1.level() << " (starts at "
              << preset.levels().input_level << ")" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operators.bit_bootstrapping(C1, boot_keys);
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
