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
    // Describe what the computation needs and let HEParams choose the
    // encryption parameters. This example rescales twice, so it needs a
    // multiplicative depth of 2.
    heongpu::HEParams<Scheme> params;
    params.multiplicative_depth = 2;

    // The scale is 2^30. Every rescaling divides by a Q prime close to the
    // scale; HEParams chooses them so that the scale stays close to 2^30 at
    // every level. The first Q prime, 60 bits by default
    // (params.first_modulus_bit_size), bounds the result at the last level:
    // here 30 bits before the decimal point and about 20-30 bits after.
    params.scale_bit_size = 30;

    // Alternatives: params.security_level = heongpu::sec_level_type::sec192
    // (sec128 by default) or sec256, and params.poly_modulus_degree to choose
    // the ring size instead of the smallest one that fits.

    // HEParams takes the smallest polynomial modulus degree N whose security
    // limit fits Q and a 60 bit P prime, here 8192, and then as many 60 bit P
    // primes as fit: a larger P makes key switching faster.
    heongpu::HEParameterSet<Scheme> parameters = params.generate();
    std::cout << parameters << std::endl;

    // The parameters can also be set by hand:
    // heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>(
    //     heongpu::sec_level_type::sec128);
    // context->set_poly_modulus_degree(8192);
    // context->set_coeff_modulus_bit_sizes({60, 30, 30}, {60});
    heongpu::HEContext<Scheme> context = parameters.make_context();
    const size_t poly_modulus_degree = parameters.poly_modulus_degree;

    // Generate a HEonGPU context with these parameters. The context checks the
    // validity of the parameters and provides various utilities needed for
    // encryption, decryption, and evaluation.
    context->generate();
    context->print_parameters();

    double scale = params.scale();

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

    // Alternative: HostVector use use pinned memory and memory pool, provide
    // faster data transfer between CPU and GPU.(and vice versa)
    // heongpu::HostVector<double> message(slot_count, 3);
    // message[0] = 10;
    // message[1] = 20;
    // message[2] = 30;
    // message[3] = 40;
    // message[4] = 0.5;

    //  [10,  20,  30,  40,  0.5,  3, ...,  3]

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

    std::cout << "Square message homomorphically." << std::endl;
    operators.multiply_inplace(C1, C1);
    std::cout << "Remove non-linear part of ciphertext." << std::endl;
    operators.relinearize_inplace(C1, relin_key);
    std::cout << "Divide ciphertext to last modulus and reduce noise."
              << std::endl;
    operators.rescale_inplace(C1);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext<Scheme> P2(context);
    decryptor.decrypt(P2, C1);

    std::cout << "Decode Plaintext and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<double> check1;
    encoder.decode(check1, P2);

    //  Approximately:
    //  [100,  400, 900, 1600, 0.25, 9, ...,  9]

    std::cout << "Check result:" << std::endl;
    display_vector(check1);

    std::vector<double> message2(slot_count, 0.25); // In CPU

    //  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5]

    std::cout << "Message2 plaintext vector." << std::endl;
    display_vector(message2);

    heongpu::Plaintext<Scheme> P3(context);
    encoder.encode(P3, message2, scale);

    std::cout << "Drop the P3 last modulus." << std::endl;
    operators.mod_drop_inplace(
        P3); // for now, do it manually for each levels you need to go.

    heongpu::Ciphertext<Scheme> C2(context);
    std::cout << "Multiply ciphertext with plaintext." << std::endl;
    operators.multiply_plain(C1, P3, C2);
    std::cout << "Add ciphertext to itself." << std::endl;
    operators.add_inplace(C2, C2);

    operators.rescale_inplace(C2);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext<Scheme> P4(context);
    decryptor.decrypt(P4, C2);

    std::vector<double> check2;
    encoder.decode(check2, P4);

    //  [50, 200, 450, 800, 4.5, 4.5, ..., 4.5]

    std::cout << "Check result2:" << std::endl;
    display_vector(check2);

    return EXIT_SUCCESS;
}
