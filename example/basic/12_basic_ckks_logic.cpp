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
constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // Initialize encryption parameters for the CKKS scheme.
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // Set the polynomial modulus degree. This controls the complexity
    // of the computations and the size of the ciphertexts. Larger values
    // allow more complex operations but increase computational cost.
    // Here, we choose 8192, a moderate value.
    size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Alternative -> select sec_level_type as sec128(default), sec192,
    // sec256, none
    // heongpu::HEContext<heongpu::Scheme::CKKS> context =
    // heongpu::GenHEContext<heongpu::Scheme::CKKS>(
    //    heongpu::sec_level_type::sec128);
    // Key switching splits Q into digits of as many primes as P has.

    // A recommended approach is to align the initial scale and the primes in
    // the coeff_modulus closely. After multiplication and rescaling, this
    // alignment helps maintain a consistent scale throughout computations. For
    // a circuit with depth D, D rescalings are required, removing D primes from
    // the coeff_modulus. The remaining prime should be slightly larger than the
    // scale to preserve the plaintext's precision.

    // A suggested for parameter selection: Start with a 60-bit prime in the
    // Q(ciphertext modulus) for maximum decryption precision. Use 60-bit prime
    // for P, ensuring it matches the size of the others (Q_tilda = QxP). Select
    // intermediate primes that are similar in value.
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});

    // Generate a HEonGPU context with these parameters. The context checks the
    // validity of the parameters and provides various utilities needed for
    // encryption, decryption, and evaluation.
    context->generate();
    context->print_parameters();

    // The scale is set to 2^30, resulting in 30 bits of precision before the
    // decimal point and approximately 20-30 bits after at the final level.
    // Intermediate primes, close to 30 bits, ensure consistent scale
    // stabilization.
    double scale = pow(2.0, 30);

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization.
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
    heongpu::HELogicOperator<Scheme> operators(context, encoder, scale);

    // Generate simple matrix in CPU.
    const int slot_count = poly_modulus_degree / 2;
    std::cout << "Plaintext vector size: " << slot_count << std::endl;
    std::vector<double> message(slot_count, 1.0);
    message[0] = 1.0;
    message[1] = 1.0;
    message[2] = 0.0;
    message[3] = 1.0;
    message[4] = 0.0;

    // Alternative: HostVector use use pinned memory and memory pool, provide
    // faster data transfer between CPU and GPU.(and vice versa)
    // heongpu::HostVector<double> message(slot_count, 3);
    // message[0] = 1.0;
    // message[1] = 1.0;
    // message[2] = 0.0;
    // message[3] = 1.0;
    // message[4] = 0.0;

    //  [1,  1,  0,  1,  0,  1, ...,  1]

    std::cout << "Message plaintext vector:" << std::endl;
    display_vector(message);

    std::cout << "Transfer vector to GPU and encode vector." << std::endl;
    heongpu::Plaintext<Scheme> P1(context);
    //  Transfer that vector from CPU to GPU and Encode that simple vector in
    //  GPU.
    encoder.encode(P1, message, scale);

    // Alternative way!
    // heongpu::Plaintext P1(context);
    // encoder.encode(P1, message, scale);

    // This plaintext(in GPU) value will be converted into an encrypted form
    // (ciphertext in GPU), which can be used for secure computations.
    std::cout << "Encrypt plaintext vector." << std::endl;
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "AND message homomorphically." << std::endl;
    heongpu::Ciphertext<Scheme> C2(context);
    operators.AND(C1, C1, C2, relin_key);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext<Scheme> P2(context);
    decryptor.decrypt(P2, C2);

    std::cout << "Decode Plaintext and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<double> check1;
    encoder.decode(check1, P2);

    //  Approximately:
    //  [1,  1,  0,  1,  0,  1, ...,  1]

    std::cout << "Check result:" << std::endl;
    display_vector(check1);

    std::vector<double> message2(slot_count, 1.0); // In CPU

    //  [1,  1,  1,  1,  1,  1, ...,  1]

    std::cout << "Message2 plaintext vector." << std::endl;
    display_vector(message2);

    heongpu::Plaintext<Scheme> P3(context);
    encoder.encode(P3, message2, scale);

    std::cout << "Drop the P3 last modulus." << std::endl;
    operators.mod_drop_inplace(
        P3); // for now, do it manually for each levels you need to go.

    std::cout << "XNOR ciphertext with plaintext." << std::endl;
    operators.XNOR_inplace(C2, P3);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext<Scheme> P4(context);
    decryptor.decrypt(P4, C2);

    std::vector<double> check2;
    encoder.decode(check2, P4);

    //  [1,  1,  0,  1,  0,  1, ...,  1]

    std::cout << "Check result2:" << std::endl;
    display_vector(check2);

    return EXIT_SUCCESS;
}
