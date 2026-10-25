// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>

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

TEST(HEonGPU, MemoryPool_Context_Generation_With_Little_Free_Device_Memory)
{
    // Simulates another process holding most of the GPU before the first
    // context of this process is generated with the default pool settings.
    auto& pool = heongpu::MemoryPool::instance();
    if (pool.get_free_device_pool_memory() != 0)
    {
        GTEST_SKIP() << "The memory pool was already created in this process.";
    }
    size_t free_memory = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_memory, &total_memory), cudaSuccess);
    const size_t leave_free = 1536ULL * 1024 * 1024;
    if (free_memory < 2 * leave_free)
    {
        GTEST_SKIP() << "Not enough free device memory for this test.";
    }
    void* blocker = nullptr;
    ASSERT_EQ(cudaMalloc(&blocker, free_memory - leave_free), cudaSuccess);

    {
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});
        ASSERT_NO_THROW(context->generate());
        EXPECT_NE(pool.get_free_device_pool_memory(), 0u);

        heongpu::DeviceVector<double> device_vector(count);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    EXPECT_EQ(cudaFree(blocker), cudaSuccess);
}

TEST(HEonGPU, MemoryPool_Shrinks_Initial_Size_That_Does_Not_Fit)
{
    auto& pool = heongpu::MemoryPool::instance();
    if (pool.get_free_device_pool_memory() != 0)
    {
        GTEST_SKIP() << "The memory pool was already created in this process.";
    }
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30}, {40});

    // All of the free memory cannot be reserved for the pool at once.
    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 1.0f;
    pool_config.max_device_fraction = 1.0f;
    pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
    pool_config.max_host_bytes = 256ULL * 1024 * 1024;
    // Sizes that do not fit are skipped instead of being attempted, so RMM
    // does not log allocation failures.
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(context->generate(pool_config));
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_EQ(log.find("maximum pool size exceeded"), std::string::npos) << log;

    size_t free_memory = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_memory, &total_memory), cudaSuccess);
    EXPECT_NE(pool.get_free_device_pool_memory(), 0u);
    EXPECT_LT(pool.get_free_device_pool_memory(), total_memory);

    heongpu::DeviceVector<double> device_vector(count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(HEonGPU, MemoryPool_Initial_Device_Fraction_Can_Be_Chosen_By_User)
{
    auto& pool = heongpu::MemoryPool::instance();
    if (pool.get_free_device_pool_memory() != 0)
    {
        GTEST_SKIP() << "The memory pool was already created in this process.";
    }
    size_t free_before = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total_memory), cudaSuccess);

    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30}, {40});

    // Reserve more than the default up front.
    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 80.0f;
    pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
    pool_config.max_host_bytes = 256ULL * 1024 * 1024;
    context->generate(pool_config);

    const double reserved =
        static_cast<double>(pool.get_free_device_pool_memory() +
                            pool.get_current_device_pool_memory_usage());
    EXPECT_GT(reserved, 0.7 * static_cast<double>(free_before));
    EXPECT_LT(reserved, 0.85 * static_cast<double>(free_before));
}
