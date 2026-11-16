// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_DEFINES_H
#define HEONGPU_DEFINES_H

#include <cstddef>

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

// Range of the polynomial degree
#define MAX_POLY_DEGREE 65536 // 2^16 for now!
#define MIN_POLY_DEGREE 4096 // 2^12

// Range of the bit-length of all user-defined modulus
#define MAX_USER_DEFINED_MOD_BIT_COUNT 60
#define MIN_USER_DEFINED_MOD_BIT_COUNT 30

// Range of the bit-length of all modulus
#define MAX_MOD_BIT_COUNT 61
#define MIN_MOD_BIT_COUNT 30

// Max auxiliary base count for BFV scheme
#define MAX_BSK_SIZE 64

// Max power of galois key capability, e.g., if MAX_SHIFT is 8, rotation
// capability range is between 0 and 255(2^(8 - 1))
#define MAX_SHIFT 8

// Memorypool sizes
// Fractions of the memory available when the pool is created. The pools grow
// on demand from the initial size up to the max size.
constexpr static float initial_device_memorypool_size =
    0.5f; // 50% of GPU memory
constexpr static float max_device_memorypool_size = 0.95f; // 95% of GPU memory

// The host pool starts with 100 MB; this fraction is not used by default.
constexpr static float initial_host_memorypool_size = 0.3f; // 30% of CPU memory
constexpr static float max_host_memorypool_size = 0.4f; // 40% of CPU memory

// On a GPU that shares the memory of the host, both pools take from one
// budget, a fraction of the available host memory, and the other fractions
// are of that budget. The device pool starts small and grows on demand.
constexpr static float unified_memorypool_budget = 0.8f; // 80% of CPU memory
constexpr static size_t unified_initial_device_memorypool_bytes = 64ULL << 20;
constexpr static float unified_max_device_memorypool_size = 0.75f;
constexpr static float unified_max_host_memorypool_size = 0.25f;

#endif // HEONGPU_DEFINES_H
