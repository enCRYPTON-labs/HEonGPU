// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <memory>

namespace
{
    heongpu::HEContext<heongpu::Scheme::CKKS> make_context(bool use_memory_pool)
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
        pool_config.use_memory_pool = use_memory_pool;
        context->generate(pool_config);
        return context;
    }

    constexpr size_t count = 1 << 16;
    constexpr size_t bytes = count * sizeof(double);
} // namespace

TEST(HEonGPU, MemoryPool_Tracks_Host_And_Device_Usage)
{
    auto context = make_context(true);

    auto& pool = heongpu::MemoryPool::instance();

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

TEST(HEonGPU, MemoryPool_Respects_Use_Memory_Pool_Flag)
{
    auto& pool = heongpu::MemoryPool::instance();

    auto pooled_context = make_context(true);
    const size_t host_base = pool.get_current_host_pool_memory_usage();
    const size_t device_base = pool.get_current_device_pool_memory_usage();

    auto pooled_host =
        std::make_unique<heongpu::HostVector<double>>(count, 0.0);
    auto pooled_device = std::make_unique<heongpu::DeviceVector<double>>(count);
    void* pooled_raw = pool.allocate(bytes);
    cudaDeviceSynchronize();

    const size_t host_pooled = pool.get_current_host_pool_memory_usage();
    const size_t device_pooled = pool.get_current_device_pool_memory_usage();
    EXPECT_GE(host_pooled, host_base + bytes);
    EXPECT_GE(device_pooled, device_base + 2 * bytes);

    // A context created with use_memory_pool = false disables the pool.
    auto unpooled_context = make_context(false);
    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        void* raw = pool.allocate(bytes);
        cudaDeviceSynchronize();

        EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_pooled);
        EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_pooled);

        pool.deallocate(raw, bytes);
    }
    cudaDeviceSynchronize();

    // Buffers allocated from the pool are returned to it while it is disabled.
    pooled_host.reset();
    pooled_device.reset();
    pool.deallocate(pooled_raw, bytes);
    cudaDeviceSynchronize();

    EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_base);
    EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_base);

    // Re-enabling the pool routes new allocations through it again.
    auto repooled_context = make_context(true);
    const size_t host_repooled = pool.get_current_host_pool_memory_usage();
    const size_t device_repooled = pool.get_current_device_pool_memory_usage();
    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();

        EXPECT_GE(pool.get_current_host_pool_memory_usage(),
                  host_repooled + bytes);
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_repooled + bytes);
    }
    cudaDeviceSynchronize();
}

TEST(HEonGPU, MemoryPool_Can_Be_Enabled_After_Unpooled_Context)
{
    // The first context of the process disables the pool; ctest runs each
    // test in its own process.
    auto& pool = heongpu::MemoryPool::instance();
    if (pool.get_free_device_pool_memory() != 0)
    {
        GTEST_SKIP() << "The memory pool was already created in this process.";
    }
    auto unpooled_context = make_context(false);
    EXPECT_EQ(pool.get_free_device_pool_memory(), 0u);

    auto pooled_context = make_context(true);
    EXPECT_NE(pool.get_free_device_pool_memory(), 0u);
    const size_t device_before = pool.get_current_device_pool_memory_usage();
    {
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_before + bytes);
    }
    cudaDeviceSynchronize();
}
