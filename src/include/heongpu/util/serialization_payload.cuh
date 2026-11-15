// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SERIALIZATION_PAYLOAD_H
#define HEONGPU_SERIALIZATION_PAYLOAD_H

#include "gpuntt/common/modular_arith.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <istream>
#include <ostream>

namespace heongpu
{
    namespace detail
    {
        // Values that pass through host memory at once while a payload is
        // written or read: 64 MB, whatever the size of the payload.
        constexpr std::size_t payload_chunk_values =
            (std::size_t(64) << 20) / sizeof(Data64);

        // Writes count values of device memory to os. The copy runs on
        // stream, after the work queued on it before. A sensitive payload,
        // such as a secret key, is erased from the host memory it passes
        // through.
        void
        write_device_payload(std::ostream& os, const Data64* data,
                             std::size_t count, cudaStream_t stream,
                             bool sensitive = false,
                             std::size_t chunk_values = payload_chunk_values);

        // Reads count values from is into device memory, on stream, and
        // waits for the copy. Reading stops where is ends; the values read
        // until then are copied and the caller checks the state of is.
        void
        read_device_payload(std::istream& is, Data64* data, std::size_t count,
                            cudaStream_t stream, bool sensitive = false,
                            std::size_t chunk_values = payload_chunk_values);
    } // namespace detail
} // namespace heongpu

#endif // HEONGPU_SERIALIZATION_PAYLOAD_H
