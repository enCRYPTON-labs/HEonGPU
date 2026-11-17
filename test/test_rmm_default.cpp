// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/per_device_resource.hpp>

// HEonGPU allocates from its own memory resource and leaves the process-wide
// RMM default alone, unless MemoryPoolConfig::set_as_rmm_default asks for it.

namespace
{
    constexpr size_t buffer_size = size_t(1) << 20;

    heongpu::HEContext<heongpu::Scheme::CKKS>
    generate_context(const heongpu::MemoryPoolConfig& config)
    {
        auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
        context->generate(config);
        return context;
    }

    // Whether an allocation from the RMM default comes from HEonGPU's pool.
    bool rmm_default_uses_heongpu_pool()
    {
        auto& pool = heongpu::MemoryPool::instance();
        const size_t before = pool.get_current_device_pool_memory_usage();
        rmm::device_buffer buffer(buffer_size, rmm::cuda_stream_default);
        const size_t during = pool.get_current_device_pool_memory_usage();
        return during >= before + buffer_size;
    }
} // namespace

// The memory pool is shared by the whole process, so the steps run in order.
TEST(HEonGPU, MemoryPool_Leaves_The_RMM_Default_Alone_Unless_Asked)
{
    heongpu::MemoryPoolConfig config;
    auto context = generate_context(config);
    EXPECT_FALSE(rmm_default_uses_heongpu_pool());

    config.set_as_rmm_default = true;
    auto shared_context = generate_context(config);
    EXPECT_TRUE(rmm_default_uses_heongpu_pool());

    // Turning it off restores the previous default.
    config.set_as_rmm_default = false;
    auto private_context = generate_context(config);
    EXPECT_FALSE(rmm_default_uses_heongpu_pool());
}

TEST(HEonGPU, MemoryPool_Keeps_A_Default_Set_By_The_Application)
{
    rmm::mr::cuda_memory_resource application_resource;
    rmm::mr::set_current_device_resource(
        rmm::device_async_resource_ref{application_resource});

    heongpu::MemoryPoolConfig config;
    config.set_as_rmm_default = true;
    auto shared_context = generate_context(config);
    EXPECT_TRUE(rmm_default_uses_heongpu_pool());

    config.set_as_rmm_default = false;
    auto private_context = generate_context(config);
    EXPECT_FALSE(rmm_default_uses_heongpu_pool());
    EXPECT_TRUE(rmm::mr::get_current_device_resource_ref() ==
                rmm::device_async_resource_ref{application_resource});

    rmm::mr::reset_current_device_resource();
}

// The documented way to silence RMM, the only library that HEonGPU uses that
// can log.
TEST(HEonGPU, RMM_Log_Level_Can_Be_Set)
{
    const auto level = rmm::default_logger().level();
    rmm::default_logger().set_level(rapids_logger::level_enum::off);
    EXPECT_EQ(rmm::default_logger().level(), rapids_logger::level_enum::off);
    rmm::default_logger().set_level(level);
}
