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
    ////////////// Alice Setup (Key Generation) ///////////////
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

    ///////////////////////////////////////////////////////////
    //////////////// Bob Setup (Key Generation) ///////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen_bob(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key_bob(context);
    keygen_bob.generate_secret_key(secret_key_bob);

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_bob(context);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::BFV> public_key_bob(
        context, common_seed);
    mpc_manager_bob.generate_public_key_share(public_key_bob, secret_key_bob);

    ///////////////////////////////////////////////////////////
    ///////////// Charlie Setup (Key Generation) //////////////
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

    ///////////////////////////////////////////////////////////
    //////////////// Server Setup (Key Sharing) ///////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::MultipartyPublickey<heongpu::Scheme::BFV>>
        participant_public_keys;
    participant_public_keys.push_back(public_key_alice);
    participant_public_keys.push_back(public_key_bob);
    participant_public_keys.push_back(public_key_charlie);

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> mpc_manager_server(
        context);
    heongpu::Publickey<heongpu::Scheme::BFV> common_public_key(context);
    mpc_manager_server.assemble_public_key_share(participant_public_keys,
                                                 common_public_key);

    ///////////////////////////////////////////////////////////
    ///////////////// Alice Setup (Encryption) ////////////////
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
    ////////////////// Bob Setup (Encryption) /////////////////
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
    /////////////// Charlie Setup (Encryption) ////////////////
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
    ////////// Server Setup (Homomorphic Operations) //////////
    ///////////////////////////////////////////////////////////

    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(
        context, encoder_charlie);

    heongpu::Ciphertext<heongpu::Scheme::BFV> cipher_accum(context);
    operators.add(ciphertext_alice, ciphertext_bob, cipher_accum);
    operators.add(cipher_accum, ciphertext_charlie, cipher_accum);

    ///////////////////////////////////////////////////////////
    /////////////////// Alice Setup (ColBoot) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> boot_alice_ciphertext_alice(
        context);
    mpc_manager_alice.distributed_bootstrapping_participant(
        cipher_accum, boot_alice_ciphertext_alice, secret_key_alice,
        common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Bob Setup (ColBoot) ///////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> boot_bob_ciphertext_bob(context);
    mpc_manager_bob.distributed_bootstrapping_participant(
        cipher_accum, boot_bob_ciphertext_bob, secret_key_bob, common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Charlie Setup (ColBoot) ///////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> boot_charlie_ciphertext_charlie(
        context);
    mpc_manager_charlie.distributed_bootstrapping_participant(
        cipher_accum, boot_charlie_ciphertext_charlie, secret_key_charlie,
        common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Server Setup (ColBoot) ////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEMultiPartyManager<heongpu::Scheme::BFV> boot_server(context);

    std::vector<heongpu::Ciphertext<heongpu::Scheme::BFV>> all_boot_ciphertexts;
    all_boot_ciphertexts.push_back(boot_alice_ciphertext_alice);
    all_boot_ciphertexts.push_back(boot_bob_ciphertext_bob);
    all_boot_ciphertexts.push_back(boot_charlie_ciphertext_charlie);

    heongpu::Ciphertext<heongpu::Scheme::BFV> boot_server_ciphertext(context);
    boot_server.distributed_bootstrapping_coordinator(
        all_boot_ciphertexts, cipher_accum, boot_server_ciphertext,
        common_seed);

    ///////////////////////////////////////////////////////////
    /////////// Alice Setup (Partially Decryption) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_alice(context);
    mpc_manager_alice.decrypt_partial(boot_server_ciphertext, secret_key_alice,
                                      partial_ciphertext_alice);

    ///////////////////////////////////////////////////////////
    ///////////// Bob Setup (Partially Decryption) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_bob(context);
    mpc_manager_bob.decrypt_partial(boot_server_ciphertext, secret_key_bob,
                                    partial_ciphertext_bob);

    ///////////////////////////////////////////////////////////
    /////////// Charlie Setup (Partially Decryption) //////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::BFV> partial_ciphertext_charlie(
        context);
    mpc_manager_charlie.decrypt_partial(
        boot_server_ciphertext, secret_key_charlie, partial_ciphertext_charlie);

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
