// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/defaultmodulus.hpp>

namespace heongpu
{
    namespace defaultparams
    {
        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_128bit_sec_modulus()
        {
            static const std::unordered_map<std::size_t, std::vector<Modulus64>>
                default_modulus_128{

                    {4096,
                     {Modulus64(0x800004001), Modulus64(0x800008001),
                      Modulus64(0x1000002001)}},

                    {8192,
                     {Modulus64(0x40000084001), Modulus64(0x400000b0001),
                      Modulus64(0x8000002c001), Modulus64(0x80000050001),
                      Modulus64(0x80000064001)}},

                    {16384,
                     {Modulus64(0x800000020001), Modulus64(0x8000001a8001),
                      Modulus64(0x8000001e8001), Modulus64(0x10000000d8001),
                      Modulus64(0x1000000168001), Modulus64(0x10000001a0001),
                      Modulus64(0x10000001e0001), Modulus64(0x10000002b8001),
                      Modulus64(0x10000002e8001)}},

                    {32768,
                     {Modulus64(0x2000000002b0001),
                      Modulus64(0x2000000003a0001),
                      Modulus64(0x2000000005b0001),
                      Modulus64(0x200000000640001),
                      Modulus64(0x400000000270001),
                      Modulus64(0x400000000350001),
                      Modulus64(0x400000000360001),
                      Modulus64(0x4000000004d0001),
                      Modulus64(0x400000000570001),
                      Modulus64(0x400000000660001),
                      Modulus64(0x4000000008a0001),
                      Modulus64(0x400000000920001),
                      Modulus64(0x400000000980001),
                      Modulus64(0x400000000990001),
                      Modulus64(0x400000000a40001)}},

                    {65536, {Modulus64(0x2000000003a0001),
                             Modulus64(0x200000000640001),
                             Modulus64(0x200000000f80001),
                             Modulus64(0x200000001460001),
                             Modulus64(0x2000000015a0001),
                             Modulus64(0x2000000015e0001),
                             Modulus64(0x200000001b20001),
                             Modulus64(0x200000001c00001),
                             Modulus64(0x200000001ee0001),
                             Modulus64(0x400000000360001),
                             Modulus64(0x400000000660001),
                             Modulus64(0x4000000008a0001),
                             Modulus64(0x400000000920001),
                             Modulus64(0x400000000980001),
                             Modulus64(0x400000000a40001),
                             Modulus64(0x400000000c00001),
                             Modulus64(0x400000000ea0001),
                             Modulus64(0x400000001460001),
                             Modulus64(0x400000001700001),
                             Modulus64(0x400000001740001),
                             Modulus64(0x4000000017a0001),
                             Modulus64(0x400000001920001),
                             Modulus64(0x400000001b00001),
                             Modulus64(0x400000001b60001),
                             Modulus64(0x400000001c40001),
                             Modulus64(0x400000001ee0001),
                             Modulus64(0x400000001f20001),
                             Modulus64(0x4000000020c0001),
                             Modulus64(0x400000002360001),
                             Modulus64(0x400000002480001)}}};

            return default_modulus_128;
        }

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_192bit_sec_modulus()
        {
            static const std::unordered_map<std::size_t, std::vector<Modulus64>>
                default_modulus_192{

                    {4096, {Modulus64(0x1000002001), Modulus64(0x1000042001)}},

                    {8192,
                     {Modulus64(0x100008c001), Modulus64(0x1000090001),
                      Modulus64(0x10000c8001), Modulus64(0x2000088001)}},

                    {16384,
                     {Modulus64(0x20000000b0001), Modulus64(0x2000000178001),
                      Modulus64(0x20000001a0001), Modulus64(0x2000000208001),
                      Modulus64(0x20000003b0001), Modulus64(0x20000003c8001)}},

                    {32768,
                     {Modulus64(0x40000000120001), Modulus64(0x400000001d0001),
                      Modulus64(0x400000002c0001), Modulus64(0x40000000480001),
                      Modulus64(0x40000000540001), Modulus64(0x400000005c0001),
                      Modulus64(0x400000006c0001), Modulus64(0x400000007b0001),
                      Modulus64(0x40000000890001), Modulus64(0x40000000b00001),
                      Modulus64(0x40000000e40001)}},

                    {65536, {Modulus64(0x40000000120001),
                             Modulus64(0x400000002c0001),
                             Modulus64(0x40000000480001),
                             Modulus64(0x40000000540001),
                             Modulus64(0x400000005c0001),
                             Modulus64(0x400000006c0001),
                             Modulus64(0x40000000b00001),
                             Modulus64(0x40000000e40001),
                             Modulus64(0x40000000f60001),
                             Modulus64(0x400000010a0001),
                             Modulus64(0x400000011a0001),
                             Modulus64(0x40000001200001),
                             Modulus64(0x40000001340001),
                             Modulus64(0x400000017a0001),
                             Modulus64(0x40000001c40001),
                             Modulus64(0x40000001ca0001),
                             Modulus64(0x40000001d00001),
                             Modulus64(0x40000002100001),
                             Modulus64(0x400000022a0001),
                             Modulus64(0x400000022e0001),
                             Modulus64(0x80000000080001),
                             Modulus64(0x80000000440001)}}};

            return default_modulus_192;
        }

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_256bit_sec_modulus()
        {
            static const std::unordered_map<std::size_t, std::vector<Modulus64>>
                default_modulus_256{

                    {4096, {Modulus64(0x8008001), Modulus64(0x10006001)}},

                    {8192,
                     {Modulus64(0x2000088001), Modulus64(0x20000e0001),
                      Modulus64(0x4000038001)}},

                    {16384,
                     {Modulus64(0x200000008001), Modulus64(0x2000000a0001),
                      Modulus64(0x2000000e0001), Modulus64(0x400000008001),
                      Modulus64(0x400000060001)}},

                    {32768,
                     {Modulus64(0x4000000120001), Modulus64(0x40000001b0001),
                      Modulus64(0x4000000270001), Modulus64(0x8000000110001),
                      Modulus64(0x8000000130001), Modulus64(0x80000001c0001),
                      Modulus64(0x80000002c0001), Modulus64(0x80000004d0001),
                      Modulus64(0x80000004f0001)}},

                    {65536,
                     {Modulus64(0x4000000120001), Modulus64(0x4000000420001),
                      Modulus64(0x4000000660001), Modulus64(0x40000007e0001),
                      Modulus64(0x4000000800001), Modulus64(0x40000008a0001),
                      Modulus64(0x7fffffffe0001), Modulus64(0x80000001c0001),
                      Modulus64(0x80000002c0001), Modulus64(0x8000000500001),
                      Modulus64(0x8000000820001), Modulus64(0x8000000940001),
                      Modulus64(0x8000001120001), Modulus64(0x80000012a0001),
                      Modulus64(0x8000001360001), Modulus64(0x80000014c0001),
                      Modulus64(0x8000001540001), Modulus64(0x8000001600001)}}};

            return default_modulus_256;
        }

    } // namespace defaultparams
} // namespace heongpu
