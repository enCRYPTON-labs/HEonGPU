// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SERIALIZER_H
#define HEONGPU_SERIALIZER_H

#include <vector>
#include <sstream>
#include <type_traits>
#include <ostream>
#include <istream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <stdint.h>
#include <zlib.h>

namespace heongpu
{
    namespace serializer
    {

        /**
         * @brief Convert a stringstream to a byte buffer.
         */
        std::vector<uint8_t> to_buffer(const std::stringstream& ss);

        /**
         * @brief Load a byte buffer into a stringstream.
         */
        void from_buffer(std::stringstream& ss,
                         const std::vector<uint8_t>& buffer);

        /**
         * @brief Compress raw data using zlib.
         * @throws std::runtime_error on failure.
         */
        std::vector<uint8_t> compress(const std::vector<uint8_t>& data);

        /**
         * @brief Decompress zlib-compressed data.
         * @throws std::runtime_error on failure.
         */
        std::vector<uint8_t> decompress(const std::vector<uint8_t>& data);

        /**
         * @brief Trait to detect serializable types (having save/load methods).
         */
        template <typename, typename = void>
        struct is_serializable : std::false_type
        {
        };

        template <typename T>
        struct is_serializable<
            T, std::void_t<decltype(std::declval<const T>().save(
                               std::declval<std::ostream&>())),
                           decltype(std::declval<T>().load(
                               std::declval<std::istream&>()))>>
            : std::true_type
        {
        };

        template <typename T>
        inline constexpr bool is_serializable_v = is_serializable<T>::value;

        /**
         * @brief Serialize an object to a compressed byte buffer.
         */
        template <typename T>
        std::enable_if_t<is_serializable_v<T>, std::vector<uint8_t>>
        serialize(const T& obj)
        {
            std::stringstream ss;
            obj.save(ss);
            return compress(to_buffer(ss));
        }

        /**
         * @brief Deserialize an object from a compressed byte buffer.
         */
        template <typename T>
        std::enable_if_t<is_serializable_v<T>, T>
        deserialize(const std::vector<uint8_t>& buffer)
        {
            std::stringstream ss;
            from_buffer(ss, decompress(buffer));
            T obj;
            obj.load(ss);
            return obj;
        }

        /**
         * @brief Save a serializable object to a binary file.
         */
        template <typename T>
        void save_to_file(const T& obj, const std::string& filename)
        {
            auto data = serialize(obj);
            uint64_t size = data.size();

            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs)
                throw std::runtime_error("Cannot open file for writing: " +
                                         filename);
            ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
            ofs.write(reinterpret_cast<const char*>(data.data()), size);
        }

        /**
         * @brief Load a serializable object from a binary file.
         */
        template <typename T> T load_from_file(const std::string& filename)
        {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs)
                throw std::runtime_error("Cannot open file for reading: " +
                                         filename);

            uint64_t size;
            ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
            std::vector<uint8_t> buffer(size);
            ifs.read(reinterpret_cast<char*>(buffer.data()), size);

            return deserialize<T>(buffer);
        }

    } // namespace serializer
} // namespace heongpu
#endif // HEONGPU_SERIALIZER_H