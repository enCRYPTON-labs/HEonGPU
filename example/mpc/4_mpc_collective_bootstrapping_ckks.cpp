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
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);

    size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({60, 50, 50, 50}, {60});
    context->generate();
    context->print_parameters();

    double scale = pow(2.0, 50);

    heongpu::RNGSeed common_seed; // automatically generate itself

    std::vector<int> shift_value = {1};

    ///////////////////////////////////////////////////////////
    ////////////// Alice Setup (Key Generation) ///////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen_alice(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key_alice(context);
    keygen_alice.generate_secret_key(secret_key_alice);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_alice(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager_alice(
        context, encoder_alice, scale);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::CKKS> public_key_alice(
        context, common_seed);
    mpc_manager_alice.generate_public_key_share(public_key_alice,
                                                secret_key_alice);

    ///////////////////////////////////////////////////////////
    //////////////// Bob Setup (Key Generation) ///////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen_bob(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key_bob(context);
    keygen_bob.generate_secret_key(secret_key_bob);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_bob(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager_bob(
        context, encoder_bob, scale);

    // Publickey
    heongpu::MultipartyPublickey<heongpu::Scheme::CKKS> public_key_bob(
        context, common_seed);
    mpc_manager_bob.generate_public_key_share(public_key_bob, secret_key_bob);

    ///////////////////////////////////////////////////////////
    ///////////// Charlie Setup (Key Generation) //////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen_charlie(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key_charlie(context);
    keygen_charlie.generate_secret_key(secret_key_charlie);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_charlie(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager_charlie(
        context, encoder_charlie, scale);

    heongpu::MultipartyPublickey<heongpu::Scheme::CKKS> public_key_charlie(
        context, common_seed);
    mpc_manager_charlie.generate_public_key_share(public_key_charlie,
                                                  secret_key_charlie);

    ///////////////////////////////////////////////////////////
    //////////////// Server Setup (Key Sharing) ///////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::MultipartyPublickey<heongpu::Scheme::CKKS>>
        participant_public_keys;
    participant_public_keys.push_back(public_key_alice);
    participant_public_keys.push_back(public_key_bob);
    participant_public_keys.push_back(public_key_charlie);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_server(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager_server(
        context, encoder_alice, scale);
    heongpu::Publickey<heongpu::Scheme::CKKS> common_public_key(context);
    mpc_manager_server.assemble_public_key_share(participant_public_keys,
                                                 common_public_key);

    ///////////////////////////////////////////////////////////
    ///////////////// Alice Setup (Encryption) ////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_alice(
        context, common_public_key);

    const int slot_count = poly_modulus_degree / 2;
    std::vector<double> message_alice(slot_count, 3.0);
    message_alice[0] = 1.0;
    message_alice[1] = 10.0;
    message_alice[2] = 100.0;

    display_vector(message_alice);

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_alice(context);
    encoder_alice.encode(plaintext_alice, message_alice, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> ciphertext_alice(context);
    encryptor_alice.encrypt(ciphertext_alice, plaintext_alice);

    ///////////////////////////////////////////////////////////
    ////////////////// Bob Setup (Encryption) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_bob(
        context, common_public_key);

    // Generate simple matrix in CPU.
    std::vector<double> message_bob(slot_count, 4.0);
    message_bob[0] = 1.0;
    message_bob[1] = 10.0;
    message_bob[2] = 100.0;

    display_vector(message_bob);

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_bob(context);
    encoder_bob.encode(plaintext_bob, message_bob, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> ciphertext_bob(context);
    encryptor_bob.encrypt(ciphertext_bob, plaintext_bob);

    ///////////////////////////////////////////////////////////
    /////////////// Charlie Setup (Encryption) ////////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_charlie(
        context, common_public_key);

    // Generate simple matrix in CPU.
    std::vector<double> message_charlie(slot_count, 5.0);
    message_charlie[0] = 1.0;
    message_charlie[1] = 10.0;
    message_charlie[2] = 100.0;

    display_vector(message_charlie);

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_charlie(context);
    encoder_charlie.encode(plaintext_charlie, message_charlie, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> ciphertext_charlie(context);
    encryptor_charlie.encrypt(ciphertext_charlie, plaintext_charlie);

    ///////////////////////////////////////////////////////////
    ////////// Server Setup (Homomorphic Operations) //////////
    ///////////////////////////////////////////////////////////

    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(
        context, encoder_charlie);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> accum_cipher(context);
    operators.add(ciphertext_alice, ciphertext_bob, accum_cipher);
    operators.add(accum_cipher, ciphertext_charlie, accum_cipher);
    operators.mod_drop_inplace(accum_cipher);
    operators.mod_drop_inplace(accum_cipher);
    std::cout << "Current Depth: " << accum_cipher.depth()
              << " (Before Collective Bootstrapping)" << std::endl;

    ///////////////////////////////////////////////////////////
    /////////////////// Alice Setup (ColBoot) /////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_alice_ciphertext_alice(
        context);
    mpc_manager_alice.distributed_bootstrapping_participant(
        accum_cipher, boot_alice_ciphertext_alice, secret_key_alice,
        common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Bob Setup (ColBoot) ///////////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_bob_ciphertext_bob(context);
    mpc_manager_bob.distributed_bootstrapping_participant(
        accum_cipher, boot_bob_ciphertext_bob, secret_key_bob, common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Charlie Setup (ColBoot) ///////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_charlie_ciphertext_charlie(
        context);
    mpc_manager_charlie.distributed_bootstrapping_participant(
        accum_cipher, boot_charlie_ciphertext_charlie, secret_key_charlie,
        common_seed);

    ///////////////////////////////////////////////////////////
    /////////////////// Server Setup (ColBoot) ////////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>
        all_boot_ciphertexts;
    all_boot_ciphertexts.push_back(boot_alice_ciphertext_alice);
    all_boot_ciphertexts.push_back(boot_bob_ciphertext_bob);
    all_boot_ciphertexts.push_back(boot_charlie_ciphertext_charlie);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_server_ciphertext(context);
    mpc_manager_server.distributed_bootstrapping_coordinator(
        all_boot_ciphertexts, accum_cipher, boot_server_ciphertext,
        common_seed);
    std::cout << "Current Depth: " << boot_server_ciphertext.depth()
              << " (After Collective Bootstrapping)" << std::endl;

    ///////////////////////////////////////////////////////////
    /////////// Alice Setup (Partially Decryption) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> partial_ciphertext_alice(
        context);
    mpc_manager_alice.decrypt_partial(
        boot_server_ciphertext, secret_key_alice,
        partial_ciphertext_alice); // ciphertext_alice

    ///////////////////////////////////////////////////////////
    ///////////// Bob Setup (Partially Decryption) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> partial_ciphertext_bob(context);
    mpc_manager_bob.decrypt_partial(boot_server_ciphertext, secret_key_bob,
                                    partial_ciphertext_bob); // ciphertext_alice

    ///////////////////////////////////////////////////////////
    /////////// Charlie Setup (Partially Decryption) //////////
    ///////////////////////////////////////////////////////////

    heongpu::Ciphertext<heongpu::Scheme::CKKS> partial_ciphertext_charlie(
        context);
    mpc_manager_charlie.decrypt_partial(
        boot_server_ciphertext, secret_key_charlie,
        partial_ciphertext_charlie); // ciphertext_alice

    ///////////////////////////////////////////////////////////

    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> partial_ciphertexts;
    partial_ciphertexts.push_back(partial_ciphertext_alice);
    partial_ciphertexts.push_back(partial_ciphertext_bob);
    partial_ciphertexts.push_back(partial_ciphertext_charlie);

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_result(context);
    mpc_manager_alice.decrypt(partial_ciphertexts, plaintext_result);

    std::vector<double> check_result;
    encoder_alice.decode(check_result, plaintext_result);

    display_vector(check_result);

    return EXIT_SUCCESS;
}
