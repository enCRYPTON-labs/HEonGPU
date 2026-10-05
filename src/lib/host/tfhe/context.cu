// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/context.cuh>

namespace heongpu
{
    HEContextImpl<Scheme::TFHE>::HEContextImpl()
        : HEContextImpl(MemoryPoolConfig::Defaults())
    {
    }

    HEContextImpl<Scheme::TFHE>::HEContextImpl(
        const MemoryPoolConfig& pool_config)
    {
        // Memory pool initialization
        MemoryPool::instance().initialize(pool_config);
        MemoryPool::instance().use_memory_pool(pool_config.use_memory_pool);
        cudaDeviceSynchronize();

        prime_ = Modulus64(1152921504606877697ULL);
        Data64 psi = 1689264667710614ULL;
        Data64 psi_inv = OPERATOR64::modinv(psi, prime_);

        std::vector<Root64> forward_table = compute_ntt_table(psi, prime_, 10);
        std::vector<Root64> inverse_table =
            compute_ntt_table(psi_inv, prime_, 10);

        ntt_table_ = std::make_shared<DeviceVector<Root64>>(forward_table);
        intt_table_ = std::make_shared<DeviceVector<Root64>>(inverse_table);

        n_inverse_ = OPERATOR64::modinv(1024, prime_);

        ks_base_bit_ = 2;
        ks_length_ = 8;

        double sqrt_two_over_pi = std::sqrt(2.0 / M_PI);
        ks_stdev_ = (1.0 / 32768.0) * sqrt_two_over_pi;
        bk_stdev_ = (9e-9) * sqrt_two_over_pi;
        max_stdev_ = (1.0 / 64.0) * sqrt_two_over_pi;

        n_ = 512;

        N_ = 1024;
        k_ = 1;

        bk_l_ = 2;
        bk_bg_bit_ = 10;
        bg_ = 1 << bk_bg_bit_;
        half_bg_ = bg_ >> 1;
        mask_mod_ = bg_ - 1;
        kpl_ = (k_ + 1) * bk_l_;
        h_ = compute_h(bk_l_, bk_bg_bit_);
        offset_ = compute_offset(bk_l_, bk_bg_bit_, half_bg_);
    }

    std::vector<int> HEContextImpl<Scheme::TFHE>::compute_h(int l, int bg_bit)
    {
        std::vector<int> h(l);
        for (int i = 0; i < l; ++i)
        {
            int shift = 32 - (i + 1) * bg_bit;
            h[i] = static_cast<int>(1) << shift;
        }
        return h;
    }

    int HEContextImpl<Scheme::TFHE>::compute_offset(int l, int bg_bit,
                                                    int half_bg)
    {
        int64_t sum = 0;
        for (int i = 1; i <= l; ++i)
        {
            int shift = 32 - i * bg_bit;
            sum += static_cast<int64_t>(1) << shift;
        }
        int64_t result = sum * half_bg;
        return static_cast<int>(result);
    }

    std::vector<Root64>
    HEContextImpl<Scheme::TFHE>::compute_ntt_table(Data64 psi, Modulus64 primes,
                                                   int n_power)
    {
        int n = 1 << n_power;
        std::vector<Root64> forward_table; // bit reverse order

        std::vector<Root64> table;
        table.push_back(1);

        for (int j = 1; j < n; j++)
        {
            Data64 exp = OPERATOR64::mult(table[(j - 1)], psi, primes);
            table.push_back(exp);
        }

        for (int j = 0; j < n; j++) // take bit reverse order
        {
            forward_table.push_back(table[gpuntt::bitreverse(j, n_power)]);
        }

        return forward_table;
    }

    template class HEContextImpl<Scheme::TFHE>;

} // namespace heongpu
