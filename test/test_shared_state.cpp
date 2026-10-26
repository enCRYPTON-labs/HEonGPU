// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

// Two libraries that use a shared HEonGPU, and the program that loads them,
// share the process-wide HEonGPU state. With a static HEonGPU, each library
// would carry its own copy.

extern "C" const void* plugin_a_memory_pool();
extern "C" const void* plugin_b_memory_pool();
extern "C" const void* plugin_a_random_number_generator();
extern "C" const void* plugin_b_random_number_generator();

TEST(HEonGPU, Shared_Library_Users_Share_The_Memory_Pool)
{
    const void* program = &heongpu::MemoryPool::instance();
    EXPECT_EQ(plugin_a_memory_pool(), program);
    EXPECT_EQ(plugin_b_memory_pool(), program);
}

TEST(HEonGPU, Shared_Library_Users_Share_The_Random_Number_Generator)
{
    const void* program = &heongpu::RandomNumberGenerator::instance();
    EXPECT_EQ(plugin_a_random_number_generator(), program);
    EXPECT_EQ(plugin_b_random_number_generator(), program);
}
