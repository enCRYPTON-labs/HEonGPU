// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/runtime.cuh>
#include <heongpu/util/random.cuh>
#include <openssl/rand.h>
#include <vector>

namespace heongpu
{
    namespace detail
    {
        void initialize_memory_pool(const MemoryPoolConfig& config)
        {
            MemoryPool::instance().initialize(config);
            MemoryPool::instance().use_memory_pool(config.use_memory_pool);
            MemoryPool::instance().set_as_rmm_default(
                config.set_as_rmm_default);
            cudaDeviceSynchronize();
        }
    } // namespace detail

    void initialize(const MemoryPoolConfig& config)
    {
        // For kernel stack size. Reserved before the memory pool so that the
        // pool does not take the memory the stack needs.
        ensure_kernel_stack_size(2048, config.raise_kernel_stack_size);

        detail::initialize_memory_pool(config);

        // DRNG initialization
        std::vector<unsigned char> generated_entropy(16); // for 128 bit
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw SystemError("RAND_bytes failed! OpenSSL could not "
                              "generate random bytes.");
        std::vector<unsigned char> generated_nonce(8); // for 128 bit
        if (1 != RAND_bytes(generated_nonce.data(), generated_nonce.size()))
            throw SystemError("RAND_bytes failed! OpenSSL could not "
                              "generate random bytes.");
        std::vector<unsigned char> personalization_string = {};
        RandomNumberGenerator::instance().initialize(
            generated_entropy, generated_nonce, personalization_string,
            rngongpu::SecurityLevel::AES128, false);
        cudaDeviceSynchronize();
    }

    void shutdown()
    {
        cudaDeviceSynchronize();
        // Throws, before anything is released, while objects use the pool.
        MemoryPool::instance().release();
        RandomNumberGenerator::instance().release();
    }

} // namespace heongpu
