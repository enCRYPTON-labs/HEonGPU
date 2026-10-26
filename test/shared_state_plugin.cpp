// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// A library that uses HEonGPU, built twice (as plugin a and plugin b) by the
// shared state test. HEONGPU_PLUGIN names the functions of each build.

#include <heongpu/heongpu.hpp>

#define HEONGPU_PLUGIN_CONCAT(a, b) a##b
#define HEONGPU_PLUGIN_FUNCTION(plugin, name)                                  \
    HEONGPU_PLUGIN_CONCAT(plugin, name)

extern "C" const void* HEONGPU_PLUGIN_FUNCTION(HEONGPU_PLUGIN, _memory_pool)()
{
    return &heongpu::MemoryPool::instance();
}

extern "C" const void* HEONGPU_PLUGIN_FUNCTION(HEONGPU_PLUGIN,
                                               _random_number_generator)()
{
    return &heongpu::RandomNumberGenerator::instance();
}
