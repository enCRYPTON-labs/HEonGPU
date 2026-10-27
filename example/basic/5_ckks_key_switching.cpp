// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <omp.h>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // Initialize encryption parameters for the CKKS scheme.
    heongpu::HEContext<Scheme> context =
        heongpu::GenHEContext<Scheme>(heongpu::sec_level_type::none);

    size_t poly_modulus_degree = 16384;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({60, 36, 36, 36, 36, 36, 36, 36},
                                         {60, 60});
    context->generate();
    context->print_parameters();

    double scale = pow(2.0, 36);

    // Generate keys: the public key for encryption, the secret key for
    // decryption, and evaluation keys(relinkey, galoiskey, switchkey).
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::Galoiskey<Scheme> galois_key(context);
    keygen.generate_galois_key(
        galois_key, secret_key); // This way will create 16(2x8) different power
                                 // of 2, if you need more change from define.h

    // Alternative way 1 -> calculate dedicated shift value:
    // std::vector<int> shifts = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, shifts);
    // keygen.generate_galois_key(galois_key, secret_key);

    // Alternative way 2 -> calculate dedicated galois value:
    // std::vector<uint32_t> galois = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, galois);
    // keygen.generate_galois_key(galois_key, secret_key);
    // use apply_galois instead of rotate_row!

    heongpu::Secretkey<Scheme> secret_key2(context);
    keygen.generate_secret_key(secret_key2);

    heongpu::Switchkey<Scheme> switch_key1_to_key2(context);
    keygen.generate_switch_key(
        switch_key1_to_key2, secret_key2, secret_key,
        heongpu::ExecutionOptions().set_storage_type(
            heongpu::storage_type::host)); // generate_switch_key is stored in
                                           // HOST

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    // Initialize 2 different Decryptor with respect to different secretkeys.
    heongpu::HEDecryptor<Scheme> decryptor_key1(context, secret_key);
    heongpu::HEDecryptor<Scheme> decryptor_key2(context, secret_key2);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // Generate simple matrix in CPU.
    const int slot_count = poly_modulus_degree / 2;
    std::cout << "Plaintext vector size: " << slot_count << std::endl;
    std::vector<double> message(slot_count, 3);
    message[0] = 10;
    message[1] = 20;
    message[2] = 30;
    message[3] = 40;
    message[4] = 0.5;

    //  [10,  20,  30,  40,  0.5,  3, ...,  3]

    std::cout << "Message plaintext vector:" << std::endl;
    display_vector(message);

    std::cout << "Transfer vector to GPU and encode vector." << std::endl;
    heongpu::Plaintext<Scheme> P1(context);
    // Transfer that vector from CPU to GPU and Encode that simple vector in
    // GPU.
    encoder.encode(P1, message, scale);

    // Alternative way!
    // heongpu::Plaintext P1(context);
    // encoder.encode(P1, message, scale);

    // This plaintext(in GPU) value will be converted into an encrypted form
    // (ciphertext in GPU), which can be used for secure computations.
    std::cout << "Encrypt plaintext vector." << std::endl;
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "Square message homomorphically." << std::endl;
    operators.multiply_inplace(C1, C1);
    std::cout << "Remove non-linear part of ciphertext." << std::endl;
    operators.relinearize_inplace(C1, relin_key);
    std::cout << "Divide ciphertext to last modulus and reduce noise."
              << std::endl;
    operators.rescale_inplace(C1);

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext<Scheme> P2(context);
    decryptor_key1.decrypt(P2, C1);

    std::cout << "Decode plaintext and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<double> check1;
    encoder.decode(check1, P2);

    //  Approximately:
    //  [100,  400, 900, 1600, 0.25, 9, ...,  9]

    std::cout << "Check result:" << std::endl;
    display_vector(check1);

    std::cout << "Shift +3:" << std::endl;
    operators.rotate_inplace(C1, galois_key, 3);

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext<Scheme> P3(context);
    decryptor_key1.decrypt(P3, C1);

    std::vector<double> check2;
    encoder.decode(check2, P3);

    // Approximately:
    // [ 1600, 0.250, 9, 9, ..., 9, 100, 400, 900 ]

    std::cout << "Check result2:" << std::endl;
    display_vector(check2);

    std::cout << "Change ciphertext key, from secretkey1 to secretkey2:"
              << std::endl;
    operators.keyswitch(C1, C1, switch_key1_to_key2);

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext<Scheme> P4(context);
    decryptor_key2.decrypt(
        P4, C1); // do not forget this should be decryptor of key2

    std::vector<double> check3;
    encoder.decode(check3, P4);

    // Approximately:
    // [ 1600, 0.250, 9, 9, ..., 9, 100, 400, 900 ]

    std::cout << "Check check3:" << std::endl;
    display_vector(check3);

    return EXIT_SUCCESS;
}
