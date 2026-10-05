// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

// These examples have been developed with reference to the Microsoft SEAL
// library.

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

int main(int argc, char* argv[])
{
    // Initialize encryption parameters for the CKKS scheme.
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // Set the polynomial modulus degree. Larger values allow deeper
    // computations but increase memory use and runtime.
    size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Set coefficient modulus sizes for CKKS.
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});

    // Configure the memory pool before generating the context-> The pool size
    // can be set either by percentage (0.0-1.0) or by percentage (0-100), or by
    // absolute bytes. If a field is not set, defaults are used.
    heongpu::MemoryPoolConfig pool_config;

    // Device pool: set as a percentage of available GPU memory.
    pool_config.initial_device_fraction = 80.0f; // %80
    pool_config.max_device_fraction = 90.0f; // %90

    // Host pool: set initial size in bytes, max size by percentage.
    pool_config.initial_host_bytes = 256ULL * 1024 * 1024; // 256 MB
    pool_config.max_host_fraction = 50.0f; // %50

    // Optional: disable memory pool usage (falls back to raw CUDA allocations).
    // pool_config.use_memory_pool = false;

    // Generate the context using the custom memory pool configuration.
    context->generate(pool_config);
    context->print_parameters();

    // Print memory pool statistics (current usage and total pool sizes).
    heongpu::MemoryPool::instance().print_memory_pool_status();

    // NOTE: MemoryPool is a singleton. The first initialize() call wins.
    // Subsequent context generations will reuse the existing pool.

    return 0;
}
