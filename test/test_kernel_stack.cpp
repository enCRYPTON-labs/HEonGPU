// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

// generate() raises a CUDA kernel stack size limit below the 2048 bytes
// HEonGPU needs, unless MemoryPoolConfig::raise_kernel_stack_size is off.

namespace
{
    using heongpu::CudaException;

    void generate(const heongpu::MemoryPoolConfig& config)
    {
        auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
        context->generate(config);
    }

    void set_stack_size(size_t bytes)
    {
        HEONGPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, bytes));
    }

    size_t stack_size()
    {
        size_t bytes = 0;
        HEONGPU_CUDA_CHECK(cudaDeviceGetLimit(&bytes, cudaLimitStackSize));
        return bytes;
    }
} // namespace

TEST(HEonGPU, Kernel_Stack_Limit_That_Is_Too_Low_Is_Reported_When_Not_Raised)
{
    set_stack_size(1024);
    heongpu::MemoryPoolConfig config;
    config.raise_kernel_stack_size = false;
    EXPECT_THROW(generate(config), heongpu::StateError);
    EXPECT_EQ(stack_size(), 1024u);
}

TEST(HEonGPU, Kernel_Stack_Limit_Of_The_Application_Is_Kept)
{
    set_stack_size(4096);
    heongpu::MemoryPoolConfig config;
    config.raise_kernel_stack_size = false;
    EXPECT_NO_THROW(generate(config));
    EXPECT_EQ(stack_size(), 4096u);

    // A larger limit is not lowered either when HEonGPU may raise it.
    EXPECT_NO_THROW(generate(heongpu::MemoryPoolConfig()));
    EXPECT_EQ(stack_size(), 4096u);
}

TEST(HEonGPU, Kernel_Stack_Limit_Is_Raised_By_Default)
{
    set_stack_size(1024);
    EXPECT_NO_THROW(generate(heongpu::MemoryPoolConfig()));
    EXPECT_GE(stack_size(), 2048u);
}
