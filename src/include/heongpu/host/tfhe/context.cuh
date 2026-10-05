// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_CONTEXT_H
#define HEONGPU_TFHE_CONTEXT_H

#include <heongpu/util/util.cuh>
#include <heongpu/util/schemes.h>
#include <heongpu/util/devicevector.cuh>
#include <heongpu/util/hostvector.cuh>
#include <heongpu/util/secstdparams.h>
#include <heongpu/util/defaultmodulus.hpp>
#include <heongpu/util/random.cuh>
#include <heongpu/util/memorypool.cuh>
#include <gmp.h>
#include <heongpu/kernel/contextpool.hpp>
#include <ostream>
#include <istream>

namespace heongpu
{
    template <> class HEContextImpl<Scheme::TFHE>
    {
        template <Scheme S> friend class Secretkey;
        template <Scheme S> friend class Bootstrappingkey;
        template <Scheme S> friend class Ciphertext;
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HELogicOperator;

      public:
        HEContextImpl();
        HEContextImpl(const MemoryPoolConfig& pool_config);

        // HEContextImpl() = default;

      private:
        const scheme_type scheme_ = scheme_type::tfhe;
        const sec_level_type sec_level_ = sec_level_type::sec128;

        Modulus64 prime_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        Ninverse64 n_inverse_;

        int ks_base_bit_;
        int ks_length_;

        double ks_stdev_;
        double bk_stdev_;
        double max_stdev_;

        // LWE Context
        int n_;
        // alpha_min = ks_stdev_
        // alpha_max = max_stdev_

        // TLWE Context
        int N_; // a power of 2: degree of the polynomials
        int k_; // number of polynomials in the mask
        // alpha_min = bk_stdev_
        // alpha_max = max_stdev_
        // extracted_lwe_params

        // TGSW Context
        int bk_l_; // l
        int bk_bg_bit_; // bg_bit
        int bg_; // decomposition base (must be a power of 2)
        int half_bg_;
        int mask_mod_;
        // tlwe_params = TLWE Context
        int kpl_; // number of rows = (k+1)*l
        std::vector<int> h_;
        int offset_;

      private:
        std::vector<int32_t> compute_h(int l, int bg_bit);

        int32_t compute_offset(int l, int bg_bit, int half_bg);

        std::vector<Root64> compute_ntt_table(Data64 psi, Modulus64 primes,
                                              int n_power);
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_CONTEXT_H
