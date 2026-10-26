// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_RUNTIME_H
#define HEONGPU_RUNTIME_H

#include <heongpu/util/memorypool.cuh>

namespace heongpu
{
    /**
     * @brief Sets up the process-wide state of HEonGPU: the memory pool, the
     * random number generator and the CUDA kernel stack size limit (see the
     * "Process-Wide State" documentation).
     *
     * HEContext::generate() does this on its own, with its MemoryPoolConfig,
     * so calling initialize() is optional. It lets an application choose when
     * this happens. Calling it again keeps the existing memory pool; its
     * sizes only change after shutdown().
     */
    void
    initialize(const MemoryPoolConfig& config = MemoryPoolConfig::Defaults());

    /**
     * @brief Releases the process-wide state of HEonGPU: the memory pool and
     * the random number generator, and restores the RMM default resource if
     * MemoryPoolConfig::set_as_rmm_default replaced it.
     *
     * All HEonGPU objects (contexts, keys, plaintexts, ciphertexts, encoders,
     * encryptors, decryptors and operators) must be destroyed first;
     * otherwise StateError is thrown and nothing is released. Afterwards,
     * initialize() or HEContext::generate() set HEonGPU up again, possibly
     * with another configuration. Does nothing if HEonGPU is not set up.
     */
    void shutdown();

    namespace detail
    {
        // The memory pool part of initialize(), which the TFHE context uses.
        void initialize_memory_pool(const MemoryPoolConfig& config);
    } // namespace detail

} // namespace heongpu
#endif // HEONGPU_RUNTIME_H
