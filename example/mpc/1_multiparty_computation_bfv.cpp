// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <omp.h>

int main(int argc, char* argv[])
{
    heongpu::HEContext<heongpu::Scheme::BFV> context =
        heongpu::GenHEContext<heongpu::Scheme::BFV>();

    size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_default_values(1);
    int plain_modulus = 1032193;
    context->set_plain_modulus(plain_modulus);
    context->generate();
    context->print_parameters();

    heongpu::RNGSeed common_seed; // automatically generate itself

    std::vector<int> shift_value = {1};

    ///////////////////////////////////////////////////////////
    ///////////// Alice Setup (Stage 1) (Phases 1) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen_alice(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key_alice(context);
    keygen_alice.generate_secret_key(secret_key_alice);

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_alice(
        context);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::BFV> public_key_alice(
        context, common_seed);
    mpc_manager_alice.generate_public_key_share(public_key_alice,
                                                secret_key_alice);

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_alice_stage1(
        context, common_seed);
    mpc_manager_alice.generate_relin_key_init(relin_key_alice_stage1,
                                              secret_key_alice);

    // Galoiskey
    heongpu::MultipartyGaloiskey<heongpu::Scheme::BFV> galois_key_alice(
        context, shift_value, common_seed);
    mpc_manager_alice.generate_galois_key_share(galois_key_alice,
                                                secret_key_alice);

    ///////////////////////////////////////////////////////////
    ////////////// Bob Setup (Stage 1) (Phases 1) /////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen_bob(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key_bob(context);
    keygen_bob.generate_secret_key(secret_key_bob);

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_bob(context);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::BFV> public_key_bob(
        context, common_seed);
    mpc_manager_bob.generate_public_key_share(public_key_bob, secret_key_bob);

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_bob_stage1(
        context, common_seed);
    mpc_manager_bob.generate_relin_key_init(relin_key_bob_stage1,
                                            secret_key_bob);

    // Galoiskey
    heongpu::MultipartyGaloiskey<heongpu::Scheme::BFV> galois_key_bob(
        context, shift_value, common_seed);
    mpc_manager_bob.generate_galois_key_share(galois_key_bob, secret_key_bob);

    ///////////////////////////////////////////////////////////
    /////////// Charlie Setup (Stage 1) (Phases 1) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen_charlie(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key_charlie(context);
    keygen_charlie.generate_secret_key(secret_key_charlie);

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_charlie(
        context);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::BFV> public_key_charlie(
        context, common_seed);
    mpc_manager_charlie.generate_public_key_share(public_key_charlie,
                                                  secret_key_charlie);

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_charlie_stage1(
        context, common_seed);
    mpc_manager_charlie.generate_relin_key_init(relin_key_charlie_stage1,
                                                secret_key_charlie);

    // Galoiskey
    heongpu::MultipartyGaloiskey<heongpu::Scheme::BFV> galois_key_charlie(
        context, shift_value, common_seed);
    mpc_manager_charlie.generate_galois_key_share(galois_key_charlie,
                                                  secret_key_charlie);

    ///////////////////////////////////////////////////////////
    ///////////// Key Sharing (Stage 1) (Phases 1) ////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::MultipartyPublickey<heongpu::Scheme::BFV>>
        participant_public_keys;
    participant_public_keys.push_back(public_key_alice);
    participant_public_keys.push_back(public_key_bob);
    participant_public_keys.push_back(public_key_charlie);

    std::vector<heongpu::MultipartyRelinkey<heongpu::Scheme::BFV>>
        participant_relin_keys_stage1;
    participant_relin_keys_stage1.push_back(relin_key_alice_stage1);
    participant_relin_keys_stage1.push_back(relin_key_bob_stage1);
    participant_relin_keys_stage1.push_back(relin_key_charlie_stage1);

    std::vector<heongpu::MultipartyGaloiskey<heongpu::Scheme::BFV>>
        participant_galois_keys;
    participant_galois_keys.push_back(galois_key_alice);
    participant_galois_keys.push_back(galois_key_bob);
    participant_galois_keys.push_back(galois_key_charlie);

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen_server(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_server(
        context);
    heongpu::Publickey<heongpu::Scheme::BFV> common_public_key(context);
    mpc_manager_server.assemble_public_key_share(participant_public_keys,
                                                 common_public_key);

    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> common_relin_key_stage1(
        context, common_seed);
    mpc_manager_server.assemble_relin_key_init(participant_relin_keys_stage1,
                                               common_relin_key_stage1);

    heongpu::Galoiskey<heongpu::Scheme::BFV> common_galois_key(context,
                                                               shift_value);
    mpc_manager_server.assemble_galois_key_share(participant_galois_keys,
                                                 common_galois_key);

    ///////////////////////////////////////////////////////////
    ///////////// Alice Setup (Stage 1) (Phases 2) ////////////
    ///////////////////////////////////////////////////////////

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_alice_stage2(
        context, common_seed);
    mpc_manager_alice.generate_relin_key_share(
        common_relin_key_stage1, relin_key_alice_stage2, secret_key_alice);

    ///////////////////////////////////////////////////////////
    ////////////// Bob Setup (Stage 1) (Phases 2) /////////////
    ///////////////////////////////////////////////////////////

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_bob_stage2(
        context, common_seed);
    mpc_manager_bob.generate_relin_key_share(
        common_relin_key_stage1, relin_key_bob_stage2, secret_key_bob);

    ///////////////////////////////////////////////////////////
    //////////// Charlie Setup (Stage 1) (Phases 2) ///////////
    ///////////////////////////////////////////////////////////

    // Relinkey
    heongpu::MultipartyRelinkey<heongpu::Scheme::BFV> relin_key_charlie_stage2(
        context, common_seed);
    mpc_manager_charlie.generate_relin_key_share(
        common_relin_key_stage1, relin_key_charlie_stage2, secret_key_charlie);

    ///////////////////////////////////////////////////////////
    //////////// Key Sharing (Stage 1) (Phases 2) /////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::MultipartyRelinkey<heongpu::Scheme::BFV>>
        participant_relin_keys_stage2;
    participant_relin_keys_stage2.push_back(relin_key_alice_stage2);
    participant_relin_keys_stage2.push_back(relin_key_bob_stage2);
    participant_relin_keys_stage2.push_back(relin_key_charlie_stage2);

    heongpu::Relinkey<heongpu::Scheme::BFV> common_relin_key(context);
    mpc_manager_server.assemble_relin_key_share(participant_relin_keys_stage2,
                                                common_relin_key_stage1,
                                                common_relin_key);

    ///////////////////////////////////////////////////////////
    ////////////////// Alice Setup (Stage 2) //////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncoder<heongpu::Scheme::BFV> encoder_alice(context);
    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor_alice(
        context, common_public_key);

    // Generate simple matrix in CPU.
    const int row_size = poly_modulus_degree / 2;
    std::vector<uint64_t> message_alice(poly_modulus_degree, 7ULL); // In CPU
    message_alice[0] = 1ULL;
    message_alice[1] = 10ULL;
    message_alice[2] = 100ULL;

    display_matrix(message_alice, row_size);

    heongpu::Plaintext<heongpu::Scheme::BFV> plaintext_alice(context);
    encoder_alice.encode(plaintext_alice, message_alice);

    heongpu::Ciphertext<heongpu::Scheme::BFV> ciphertext_alice(context);
    encryptor_alice.encrypt(ciphertext_alice, plaintext_alice);

    ///////////////////////////////////////////////////////////
    /////////////////// Bob Setup (Stage 2) ///////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncoder<heongpu::Scheme::BFV> encoder_bob(context);
    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor_bob(context,
                                                             common_public_key);

    // Generate simple matrix in CPU.
    std::vector<uint64_t> message_bob(poly_modulus_degree, 8ULL); // In CPU
    message_bob[0] = 2ULL;
    message_bob[1] = 20ULL;
    message_bob[2] = 200ULL;

    display_matrix(message_bob, row_size);

    heongpu::Plaintext<heongpu::Scheme::BFV> plaintext_bob(context);
    encoder_bob.encode(plaintext_bob, message_bob);

    heongpu::Ciphertext<heongpu::Scheme::BFV> ciphertext_bob(context);
    encryptor_bob.encrypt(ciphertext_bob, plaintext_bob);

    ///////////////////////////////////////////////////////////
    ///////////////// Charlie Setup (Stage 2) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncoder<heongpu::Scheme::BFV> encoder_charlie(context);
    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor_charlie(
        context, common_public_key);

    // Generate simple matrix in CPU.
    std::vector<uint64_t> message_charlie(poly_modulus_degree, 9ULL); // In CPU
    message_charlie[0] = 3ULL;
    message_charlie[1] = 30ULL;
    message_charlie[2] = 300ULL;

    display_matrix(message_charlie, row_size);

    heongpu::Plaintext<heongpu::Scheme::BFV> plaintext_charlie(context);
    encoder_charlie.encode(plaintext_charlie, message_charlie);

    heongpu::Ciphertext<heongpu::Scheme::BFV> ciphertext_charlie(context);
    encryptor_charlie.encrypt(ciphertext_charlie, plaintext_charlie);

    ///////////////////////////////////////////////////////////
    ///////////////// Server Setup (Stage 3) //////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(
        context, encoder_charlie);

    heongpu::Ciphertext<heongpu::Scheme::BFV> cipher_mult(context);
    operators.multiply(ciphertext_alice, ciphertext_bob, cipher_mult);
    operators.relinearize_inplace(cipher_mult, common_relin_key);

    heongpu::Ciphertext<heongpu::Scheme::BFV> cipher_mult_add(context);
    operators.add(cipher_mult, ciphertext_charlie, cipher_mult_add);

    heongpu::Ciphertext<heongpu::Scheme::BFV> cipher_mult_add_rotate(context);
    operators.rotate_rows(cipher_mult_add, cipher_mult_add_rotate,
                          common_galois_key, 1);

    ///////////////////////////////////////////////////////////
    /////////////////// Alice Setup (Stage 4) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_alice(context);
    mpc_manager_alice.decrypt_partial(cipher_mult_add_rotate, secret_key_alice,
                                      partial_ciphertext_alice);

    ///////////////////////////////////////////////////////////
    /////////////////// Bob Setup (Stage 4) ///////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_bob(context);
    mpc_manager_bob.decrypt_partial(cipher_mult_add_rotate, secret_key_bob,
                                    partial_ciphertext_bob);

    ///////////////////////////////////////////////////////////
    ///////////////// Charlie Setup (Stage 4) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_charlie(
        context);
    mpc_manager_charlie.decrypt_partial(
        cipher_mult_add_rotate, secret_key_charlie, partial_ciphertext_charlie);

    ///////////////////////////////////////////////////////////

    std::vector<heongpu::Ciphertext<heongpu::Scheme::BFV>> partial_ciphertexts;
    partial_ciphertexts.push_back(partial_ciphertext_alice);
    partial_ciphertexts.push_back(partial_ciphertext_bob);
    partial_ciphertexts.push_back(partial_ciphertext_charlie);

    heongpu::Plaintext<heongpu::Scheme::BFV> plaintext_result(context);
    mpc_manager_alice.decrypt(partial_ciphertexts, plaintext_result);

    std::vector<uint64_t> check_result;
    encoder_alice.decode(check_result, plaintext_result);

    display_matrix(check_result, row_size);

    return EXIT_SUCCESS;
}