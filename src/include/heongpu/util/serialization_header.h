// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SERIALIZATION_HEADER_H
#define HEONGPU_SERIALIZATION_HEADER_H

#include <heongpu/util/schemes.h>
#include <cstdint>
#include <istream>
#include <ostream>

namespace heongpu
{
    namespace detail
    {
        // Object types stored in the serialization_format::v2 header.
        enum class serialized_object : std::uint8_t
        {
            context = 0x1,
            secret_key = 0x2,
            public_key = 0x3,
            relin_key = 0x4,
            galois_key = 0x5,
            switch_key = 0x6,
            plaintext = 0x7,
            ciphertext = 0x8
        };

        // Writes the serialization_format::v2 header: the magic number
        // "HEGP", the format version, the object type, the scheme and the
        // context identity, in little-endian order. Writes nothing for
        // serialization_format::v1.
        void write_serialization_header(std::ostream& os,
                                        serialization_format format,
                                        serialized_object object,
                                        scheme_type scheme,
                                        std::uint64_t context_id);

        // Reads and checks the serialization_format::v2 header when the
        // stream starts with one, and returns the context identity stored in
        // it. Returns 0 without reading anything for serialization_format::v1
        // data, which starts with the scheme.
        std::uint64_t read_serialization_header(std::istream& is,
                                                serialized_object object,
                                                scheme_type scheme);

        // Throws ContextMismatchError when the context identity read from a
        // stream differs from the one of the object it is loaded into. An
        // identity of 0 on either side is unknown and not checked.
        void validate_loaded_context_identity(std::uint64_t object_id,
                                              std::uint64_t loaded_id,
                                              const char* object_name);

    } // namespace detail
} // namespace heongpu
#endif // HEONGPU_SERIALIZATION_HEADER_H
