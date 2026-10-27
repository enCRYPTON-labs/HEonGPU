// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <cmath>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // In the automatic rescale mode, the operators rescale the ciphertexts
    // when an operation needs it and bring operands of different levels to
    // the same level and scale. The application only relinearizes, which
    // needs the relinearization key.
    heongpu::HEParams<Scheme> params;
    params.multiplicative_depth = 3;
    params.rescale_mode = heongpu::rescale_mode::automatic;
    heongpu::HEParameterSet<Scheme> parameters = params.generate();
    std::cout << parameters << std::endl;

    // A context created by hand selects the mode after setting the moduli:
    // context->set_rescale_mode(heongpu::rescale_mode::automatic, scale);
    heongpu::HEContext<Scheme> context = parameters.make_context();
    context->generate();

    // Every level has one scale: the top level has the scale of the
    // parameters, and multiplying two ciphertexts of a level and rescaling
    // gives the scale of the next level.
    for (int level = 3; level >= 0; level--)
    {
        std::cout << "Scale of level " << level << ": 2^"
                  << std::log2(context->get_level_scale(level)) << std::endl;
    }

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key = keygen.secret_key();
    heongpu::Publickey<Scheme> public_key = keygen.public_key(secret_key);
    heongpu::Relinkey<Scheme> relin_key = keygen.relin_key(secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    const int slot_count = context->get_poly_modulus_degree() / 2;
    std::vector<double> x(slot_count), y(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        x[i] = 0.5 + 0.25 * std::sin(0.01 * i);
        y[i] = 0.25 * std::cos(0.01 * i);
    }

    // Without a scale, encode uses the scale of the top level.
    heongpu::Plaintext<Scheme> P1 = encoder.encode(x);
    heongpu::Plaintext<Scheme> P2 = encoder.encode(y);
    heongpu::Ciphertext<Scheme> X = encryptor.encrypt(P1);
    heongpu::Ciphertext<Scheme> Y = encryptor.encrypt(P2);

    // x^3 + x * y + y: no rescale and no mod_drop.
    std::cout << "Compute x^3 + x * y + y." << std::endl;
    heongpu::Ciphertext<Scheme> X2 = operators.multiply(X, X);
    operators.relinearize_inplace(X2, relin_key);
    // X2 is rescaled here, and X brought down to its level.
    heongpu::Ciphertext<Scheme> X3 = operators.multiply(X2, X);
    operators.relinearize_inplace(X3, relin_key);
    heongpu::Ciphertext<Scheme> XY = operators.multiply(X, Y);
    operators.relinearize_inplace(XY, relin_key);
    // The three terms are at different levels.
    heongpu::Ciphertext<Scheme> R = operators.add(X3, XY);
    operators.add_inplace(R, Y);
    std::cout << "Result at level " << R.level() << std::endl;

    heongpu::Plaintext<Scheme> P3 = decryptor.decrypt(R);
    std::vector<double> result = encoder.decode(P3);

    std::vector<double> expected(slot_count);
    double error = 0.0;
    for (int i = 0; i < slot_count; i++)
    {
        expected[i] = x[i] * x[i] * x[i] + x[i] * y[i] + y[i];
        error = std::max(error, std::abs(result[i] - expected[i]));
    }
    std::cout << "Expected:" << std::endl;
    display_vector(expected);
    std::cout << "Result:" << std::endl;
    display_vector(result);
    std::cout << "Maximum error: " << error << std::endl;

    return EXIT_SUCCESS;
}
