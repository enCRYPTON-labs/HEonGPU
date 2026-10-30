// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <cmath>

int main(int argc, char* argv[])
{
    // The session generates the context of the preset, the keys and the
    // bootstrapping keys up front, and works in the automatic rescale mode
    // with the level of the bootstrapping results as its top level. The
    // regular preset leaves 9 levels between two bootstrappings;
    // CKKSBootstrappingPreset::regular(levels) leaves another number.
    heongpu::HESession<heongpu::Scheme::CKKS> session(
        heongpu::CKKSBootstrappingPreset::regular());
    std::cout << session.context()->to_string() << std::endl;
    heongpu::BootstrappingLevels levels = session.bootstrapping_levels();
    std::cout << "Levels between two bootstrappings: " << levels.usable_levels()
              << std::endl;

    const int slot_count = session.slot_count();
    std::vector<double> values(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        values[i] = 0.5 * std::sin(0.001 * i);
    }

    // x <- 0.5 * x * x + 0.25 twenty times, which needs more levels than the
    // context has: the product and the product by 0.5 use a level each.
    // Bootstrapping is never started by itself; levels_left() tells when it
    // is needed.
    auto x = session.encrypt(values);
    std::vector<double> expected = values;
    int bootstraps = 0;
    for (int round = 0; round < 20; round++)
    {
        if (x.levels_left() < 2)
        {
            x = x.bootstrap();
            bootstraps++;
        }
        x = 0.5 * x * x + 0.25;
        for (double& value : expected)
        {
            value = 0.5 * value * value + 0.25;
        }
    }

    std::vector<double> result = session.decrypt(x);
    double error = 0.0;
    for (int i = 0; i < slot_count; i++)
    {
        error = std::max(error, std::abs(result[i] - expected[i]));
    }
    std::cout << "Expected:" << std::endl;
    display_vector(expected);
    std::cout << "Result:" << std::endl;
    display_vector(result);
    std::cout << "Bootstrappings: " << bootstraps
              << ", maximum error: " << error << std::endl;

    return EXIT_SUCCESS;
}
