// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/serialization_header.h>
#include <heongpu/util/error.h>
#include <array>
#include <ios>

namespace heongpu
{
    namespace detail
    {
        namespace
        {
            constexpr std::array<char, 4> magic = {'H', 'E', 'G', 'P'};
            constexpr std::uint16_t format_version = 2;
            // Version, object type, scheme and context identity.
            constexpr std::size_t field_bytes = 2 + 1 + 1 + 8;

            const char* object_name(std::uint8_t object)
            {
                switch (static_cast<serialized_object>(object))
                {
                    case serialized_object::context:
                        return "HEContext";
                    case serialized_object::secret_key:
                        return "Secretkey";
                    case serialized_object::public_key:
                        return "Publickey";
                    case serialized_object::relin_key:
                        return "Relinkey";
                    case serialized_object::galois_key:
                        return "Galoiskey";
                    case serialized_object::switch_key:
                        return "Switchkey";
                    case serialized_object::plaintext:
                        return "Plaintext";
                    case serialized_object::ciphertext:
                        return "Ciphertext";
                }
                return "unknown";
            }

            const char* scheme_name(scheme_type scheme)
            {
                switch (scheme)
                {
                    case scheme_type::bfv:
                        return "BFV";
                    case scheme_type::ckks:
                        return "CKKS";
                    case scheme_type::tfhe:
                        return "TFHE";
                    default:
                        return "unknown";
                }
            }
        } // namespace

        void write_serialization_header(std::ostream& os,
                                        serialization_format format,
                                        serialized_object object,
                                        scheme_type scheme,
                                        std::uint64_t context_id)
        {
            if (format != serialization_format::v2)
            {
                throw ParameterError(error_message(
                    "Invalid serialization format ", static_cast<int>(format),
                    "! Expected serialization_format::v2."));
            }
            std::array<unsigned char, field_bytes> fields;
            fields[0] = format_version & 0xFF;
            fields[1] = format_version >> 8;
            fields[2] = static_cast<std::uint8_t>(object);
            fields[3] = static_cast<std::uint8_t>(scheme);
            for (int i = 0; i < 8; i++)
            {
                fields[4 + i] = (context_id >> (8 * i)) & 0xFF;
            }
            os.write(magic.data(), magic.size());
            os.write(reinterpret_cast<const char*>(fields.data()),
                     fields.size());
        }

        std::uint64_t read_serialization_header(std::istream& is,
                                                serialized_object object,
                                                scheme_type scheme)
        {
            if (is.peek() != magic[0])
            {
                return 0;
            }

            std::array<char, 4> read_magic;
            is.read(read_magic.data(), read_magic.size());
            if (!is || read_magic != magic)
            {
                throw SerializationError(
                    "Invalid HEonGPU binary! The data does not start with the "
                    "HEGP magic number.");
            }

            std::array<unsigned char, field_bytes> fields;
            is.read(reinterpret_cast<char*>(fields.data()), fields.size());
            if (!is)
            {
                throw SerializationError(
                    "Truncated HEonGPU binary! The header is incomplete.");
            }

            const std::uint16_t version = fields[0] | (fields[1] << 8);
            if (version != format_version)
            {
                throw SerializationError(error_message(
                    "Unsupported HEonGPU binary format version ", version,
                    "! This HEonGPU version reads the format versions 1 and ",
                    format_version, "."));
            }
            if (fields[2] != static_cast<std::uint8_t>(object))
            {
                throw SerializationError(error_message(
                    "Invalid object binary! Expected ",
                    object_name(static_cast<std::uint8_t>(object)),
                    " data, found ", object_name(fields[2]), " data."));
            }
            if (fields[3] != static_cast<std::uint8_t>(scheme))
            {
                throw SerializationError(error_message(
                    "Invalid scheme binary! Expected ", scheme_name(scheme),
                    " data, found scheme id ", static_cast<int>(fields[3]),
                    "."));
            }

            std::uint64_t context_id = 0;
            for (int i = 0; i < 8; i++)
            {
                context_id |= std::uint64_t(fields[4 + i]) << (8 * i);
            }
            return context_id;
        }

        void validate_loaded_context_identity(std::uint64_t object_id,
                                              std::uint64_t loaded_id,
                                              const char* object_name)
        {
            if (object_id == 0 || loaded_id == 0 || object_id == loaded_id)
            {
                return;
            }
            std::ostringstream stream;
            stream << object_name << " context mismatch! The serialized "
                   << object_name << " belongs to the context with id 0x"
                   << std::hex << loaded_id << " but is loaded into an "
                   << object_name << " of the context with id 0x" << object_id
                   << ".";
            throw ContextMismatchError(stream.str());
        }

    } // namespace detail
} // namespace heongpu
