// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::BFV;

int main(int argc, char* argv[])
{
    // The session generates the context of the parameters and the keys. The
    // operators relinearize the products.
    heongpu::HEParams<Scheme> params;
    params.multiplicative_depth = 2;
    heongpu::HESession<Scheme> session(params);
    std::cout << session.context()->to_string() << std::endl;

    const int slot_count = session.slot_count();
    const uint64_t t = session.plain_modulus();
    std::vector<uint64_t> xs(slot_count), ys(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        xs[i] = i % 100;
        ys[i] = (3 * i) % 50;
    }

    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    std::cout << "Noise budget of x: " << session.noise_budget(x) << " bits"
              << std::endl;

    std::cout << "Compute x * x * y + 3 * x - y modulo " << t << "."
              << std::endl;
    auto z = x * x * y + 3 * x - y;
    std::cout << "Noise budget of the result: " << session.noise_budget(z)
              << " bits" << std::endl;
    std::vector<uint64_t> result = session.decrypt(z);

    bool correct = true;
    for (int i = 0; i < slot_count; i++)
    {
        uint64_t expected = (xs[i] * xs[i] * ys[i] + 3 * xs[i] + t - ys[i]) % t;
        correct = correct && (result[i] == expected);
    }
    std::cout << "Result:" << std::endl;
    display_vector(result);
    std::cout << "Correct: " << (correct ? "yes" : "no") << std::endl;

    // The slots form two rows: rotate() shifts both rows and
    // rotate_columns() swaps them. The Galois keys are generated the first
    // time they are used.
    auto r = x.rotate(1).rotate_columns();
    std::vector<int64_t> rotated = session.decrypt_signed(r - 50);
    std::cout << "Slot 0: " << rotated[0] << ", expected "
              << int64_t(xs[slot_count / 2 + 1]) - 50 << std::endl;

    return EXIT_SUCCESS;
}
