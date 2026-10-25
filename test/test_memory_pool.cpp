// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

TEST(HEonGPU, MemoryPool_Tracks_Host_And_Device_Usage)
{
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30}, {40});

    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_bytes = 256ULL * 1024 * 1024;
    pool_config.max_device_bytes = 1024ULL * 1024 * 1024;
    pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
    pool_config.max_host_bytes = 256ULL * 1024 * 1024;
    context->generate(pool_config);

    auto& pool = heongpu::MemoryPool::instance();
    const size_t count = 1 << 16;
    const size_t bytes = count * sizeof(double);

    const size_t host_before = pool.get_current_host_pool_memory_usage();
    const size_t device_before = pool.get_current_device_pool_memory_usage();

    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();

        EXPECT_GE(pool.get_current_host_pool_memory_usage(),
                  host_before + bytes);
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_before + bytes);
    }

    cudaDeviceSynchronize();

    EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_before);
    EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_before);
}
