// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <omp.h>

// These examples have been developed with reference to the Microsoft SEAL
// library.

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::BFV;

int main(int argc, char* argv[])
{
    // Initialize encryption parameters for the BFV scheme.
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // Set the polynomial modulus degree. This controls the complexity
    // of the computations and the size of the ciphertexts. Larger values
    // allow more complex operations but increase computational cost.
    // Here, we choose 8192, a moderate value.
    size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Alternative -> select sec_level_type as sec128(default), sec192,
    // sec256, none
    // heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>(
    //    heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
    //    heongpu::sec_level_type::sec128);

    // Set the coefficient modulus, which determines the size of the noise
    // budget. This parameter is crucial for controlling the number of
    // operations that can be performed before the ciphertext becomes too
    // noisy to decrypt. We use a default setting suitable for our chosen
    // polynomial modulus degree.
    context->set_coeff_modulus_default_values(1);

    // Alternative -> first input is Q (ciphertext modulus), second one
    // is P and it determines to Q_tilda which is key modulus.(Q_tilda = QxP)
    // context->set_coeff_modulus({40, 40, 40, 40}, {40});

    // Set the plaintext modulus, which affects the range of plaintext values
    // and the noise budget usage during multiplications. Choosing a smaller
    // value can help in maintaining a larger noise budget.
    int plain_modulus = 1032193;
    context->set_plain_modulus(plain_modulus);

    // Generate a HEonGPU context with these parameters. The context checks the
    // validity of the parameters and provides various utilities needed for
    // encryption, decryption, and evaluation.
    context->generate();
    context->print_parameters();

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization..
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Initialize Encoder, Encryptor, Evaluator, and Decryptor. The Encoder will
    // encode the message for SIMD operations. The Encryptor will use the public
    // key to encrypt data, while the Decryptor will use the secret key to
    // decrypt it. The Evaluator will handle operations on the encrypted data.
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // Generate simple matrix in CPU.
    const int row_size = poly_modulus_degree / 2;
    std::cout << "Plaintext matrix row size: " << row_size << std::endl;
    std::vector<uint64_t> message(poly_modulus_degree, 8ULL); // In CPU
    message[0] = 1ULL;
    message[1] = 12ULL;
    message[2] = 23ULL;
    message[3] = 31ULL;
    message[row_size] = 7ULL;
    message[row_size + 1] = 54ULL;
    message[row_size + 2] = 6ULL;
    message[row_size + 3] = 100ULL;

    // Alternative: HostVector use use pinned memory and memory pool, provide
    // faster data transfer between CPU and GPU.(and vice versa)
    // heongpu::HostVector<uint64_t> message(poly_modulus_degree, 8ULL); // In
    // CPU, message[0] = 1ULL; message[1] = 12ULL; message[2] = 23ULL;
    // message[3] = 31ULL;
    // message[row_size] = 7ULL;
    // message[row_size + 1] = 54ULL;
    // message[row_size + 2] = 6ULL;
    // message[row_size + 3] = 100ULL;

    //  [1,  12,  23,  31,  8,  8, ...,  8]
    //  [7,  54,  6,  100,  8,  8, ...,  8]

    std::cout << "Message plaintext matrix:" << std::endl;
    display_matrix(message, row_size);

    std::cout << "Transfer matrix to GPU and encode matrix." << std::endl;
    heongpu::Plaintext<Scheme> P1(context);
    // Transfer that vector from CPU to GPU and Encode that simple vector in
    // GPU.
    encoder.encode(P1, message);

    // Alternative way!
    // heongpu::Plaintext P1(context);
    // encoder.encode(P1, message);

    // This plaintext(in GPU) value will be converted into an encrypted form
    // (ciphertext in GPU), which can be used for secure computations.
    std::cout << "Encrypt plaintext matrix." << std::endl;
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "Initial noise budget in C1: "
              << decryptor.remainder_noise_budget(C1) << " bits" << std::endl;

    std::cout << "Square message homomorphically." << std::endl;
    operators.multiply_inplace(C1, C1);
    std::cout << "Remove non-linear part of ciphertext." << std::endl;
    operators.relinearize_inplace(C1, relin_key);

    std::cout << "Noise budget in C1 after multiplication: "
              << decryptor.remainder_noise_budget(C1) << " bits" << std::endl;

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext<Scheme> P2(context);
    decryptor.decrypt(P2, C1);

    std::cout << "Decode plaintext ing GPU and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<uint64_t> check1;
    encoder.decode(check1, P2);

    //  [1,  144,   529, 961,   64, 64, ...,  64]
    //  [49, 2916, 36,  10000, 64, 64, ...,  64]

    std::cout << "Check result:" << std::endl;
    display_matrix(check1, row_size);

    std::vector<uint64_t> message2(poly_modulus_degree, 3ULL); // In CPU

    //  [3,  3,  3,  3,  3,  3, ...,  3]
    //  [3,  3,  3,  3,  3,  3, ...,  3]

    std::cout << "Message2 plaintext matrix." << std::endl;
    display_matrix(message2, row_size);

    heongpu::Plaintext<Scheme> P3(context);
    encoder.encode(P3, message2);

    heongpu::Ciphertext<Scheme> C2(context);
    std::cout << "Mutiply ciphertext with plaintext." << std::endl;
    operators.multiply_plain(C1, P3, C2);

    // Alternative way -> select sec_level_type as sec128(default), sec192,
    // operators.transform_to_ntt_inplace(P3); // transform plaintext to ntt
    // domain, this way increase the plaintext size operators.multiply_plain(C1,
    // P3, C2); // faster than regular one

    std::cout << "Add ciphertext to itself." << std::endl;
    operators.add_inplace(C2, C2);

    std::cout << "Noise budget in C2 after multiplication: "
              << decryptor.remainder_noise_budget(C2) << " bits" << std::endl;

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext<Scheme> P4(context);
    decryptor.decrypt(P4, C2);

    std::vector<uint64_t> check2;
    encoder.decode(check2, P4);

    //  [6,   864,   3174, 5766,  384, 384, ...,  384]
    //  [294, 17496, 216,  60000, 384, 384, ...,  384]

    std::cout << "Check result2:" << std::endl;
    display_matrix(check2, row_size);

    return EXIT_SUCCESS;
}
