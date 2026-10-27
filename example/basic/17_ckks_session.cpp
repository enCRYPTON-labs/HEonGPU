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
    // The session generates the context of the parameters and the keys, and
    // works in the automatic rescale mode: the operators relinearize the
    // products, rescale them and bring operands to the same level.
    heongpu::HEParams<Scheme> params;
    params.multiplicative_depth = 3;
    heongpu::HESession<Scheme> session(params);
    std::cout << session.context()->to_string() << std::endl;

    const int slot_count = session.slot_count();
    std::vector<double> xs(slot_count), ys(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        xs[i] = 0.5 + 0.25 * std::sin(0.01 * i);
        ys[i] = 0.25 * std::cos(0.01 * i);
    }

    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);

    std::cout << "Compute x * x * y + 0.5 * x - y." << std::endl;
    auto z = x * x * y + 0.5 * x - y;
    std::vector<double> result = session.decrypt(z);

    std::vector<double> expected(slot_count);
    double error = 0.0;
    for (int i = 0; i < slot_count; i++)
    {
        expected[i] = xs[i] * xs[i] * ys[i] + 0.5 * xs[i] - ys[i];
        error = std::max(error, std::abs(result[i] - expected[i]));
    }
    std::cout << "Expected:" << std::endl;
    display_vector(expected);
    std::cout << "Result:" << std::endl;
    display_vector(result);
    std::cout << "Maximum error: " << error << std::endl;

    // Rotations generate their Galois key the first time they are used;
    // HESession(params, {1, 2}) generates the keys for the steps 1 and 2 up
    // front.
    std::cout << "Rotate x by 1 and multiply it by a vector of weights."
              << std::endl;
    std::vector<double> weights(slot_count, 2.0);
    auto w = x.rotate(1) * weights;
    std::vector<double> rotated = session.decrypt(w);
    std::cout << "Slot 0: " << rotated[0] << ", expected " << 2.0 * xs[1]
              << std::endl;

    // The ciphertext of the lower level API, and the objects of the session
    // (session.operators(), session.relin_key(), ...), stay available.
    heongpu::Ciphertext<Scheme>& ciphertext = z.ciphertext();
    std::cout << "z is at level " << ciphertext.level() << std::endl;

    return EXIT_SUCCESS;
}
