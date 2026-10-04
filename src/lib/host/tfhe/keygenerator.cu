// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/keygenerator.cuh>
#include "../secure_random.h"

namespace heongpu
{
    __host__ HEKeyGenerator<Scheme::TFHE>::HEKeyGenerator(
        HEContext<Scheme::TFHE> context)
    {
        if (!context)
        {
            throw std::invalid_argument("HEContext is not set!");
        }

        context_ = std::move(context);
    }

    __host__ void HEKeyGenerator<Scheme::TFHE>::generate_secret_key(
        Secretkey<Scheme::TFHE>& sk, const ExecutionOptions& options)
    {
        if (sk.secret_key_generated_)
        {
            throw std::runtime_error("Secretkey is already generated!");
        }

        if (options.storage_ != storage_type::DEVICE &&
            options.storage_ != storage_type::HOST)
            throw std::invalid_argument("Invalid storage type!");
        sk.validate_context(context_);
        detail::PrivateRandomBuffer<int32_t> lwe(context_->n_);
        detail::PrivateRandomBuffer<int32_t> tlwe(context_->k_ * context_->N_);
        detail::SecretKeyRandomGenerator random;
        for (auto& value : lwe.values)
            value = random() & 1U;
        for (auto& value : tlwe.values)
            value = random() & 1U;
        DeviceVector<int32_t> lwe_device(lwe.values, options.stream_);
        DeviceVector<int32_t> tlwe_device(tlwe.values, options.stream_);
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
        sk.lwe_key_device_location_ = std::move(lwe_device);
        sk.tlwe_key_device_location_ = std::move(tlwe_device);
        sk.storage_type_ = storage_type::DEVICE;
        sk.secret_key_generated_ = true;
        if (options.storage_ == storage_type::HOST)
            sk.store_in_host(options.stream_);
    }

    __host__ void HEKeyGenerator<Scheme::TFHE>::generate_bootstrapping_key(
        Bootstrappingkey<Scheme::TFHE>& bk, Secretkey<Scheme::TFHE>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!bk.context_ || bk.bk_k_ != context_->k_ ||
            bk.bk_base_bit_ != context_->bk_bg_bit_ ||
            bk.bk_length_ != context_->bk_l_ ||
            bk.bk_stdev_ != context_->bk_stdev_ ||
            bk.ks_base_bit_ != context_->ks_base_bit_ ||
            bk.ks_length_ != context_->ks_length_)
            throw std::invalid_argument("Bootstrappingkey context mismatch!");
        if (bk.boot_key_generated_)
        {
            throw std::logic_error("Bootkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::TFHE>& sk_)
            {
                output_storage_manager(
                    bk,
                    [&](Bootstrappingkey<Scheme::TFHE>& bk_)
                    {
                        // Boot Key
                        int bk_n = context_->n_;
                        int bk_N = context_->N_;
                        int bk_k = bk.bk_k_;
                        int bk_base_bit_size = bk.bk_base_bit_;
                        int bk_length = bk.bk_length_;
                        double bk_stddev = bk.bk_stdev_;

                        Data64 total_bootkey_size =
                            (Data64) bk_n * (Data64) (bk_k + 1) *
                            (Data64) bk_length * (Data64) (bk_k + 1) *
                            (Data64) bk_N;
                        DeviceVector<int32_t> temp_boot_key(total_bootkey_size,
                                                            options.stream_);

                        detail::PrivateRandomBuffer<int32_t> boot_random(
                            total_bootkey_size);
                        detail::SecretKeyRandomGenerator random;
                        std::normal_distribution<double> boot_noise(0.0,
                                                                    bk_stddev);
                        const size_t row_size =
                            static_cast<size_t>(bk_k + 1) * bk_N;
                        for (size_t row = 0; row < total_bootkey_size;
                             row += row_size)
                        {
                            detail::private_random_bytes(
                                boot_random.values.data() + row,
                                bk_k * bk_N * sizeof(int32_t));
                            for (int i = 0; i < bk_N; ++i)
                                boot_random.values[row + bk_k * bk_N + i] =
                                    detail::torus32(boot_noise(random));
                        }
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            temp_boot_key.data(), boot_random.values.data(),
                            total_bootkey_size * sizeof(int32_t),
                            cudaMemcpyHostToDevice, options.stream_));
                        HEONGPU_CUDA_CHECK(
                            cudaStreamSynchronize(options.stream_));

                        DeviceVector<Data64> tlwe_key_ntt(bk_N * bk_k,
                                                          options.stream_);
                        tfhe_convert_rlwekey_ntt_domain_kernel<<<
                            bk_k, 512, sizeof(Data64) * bk_N,
                            options.stream_>>>(
                            tlwe_key_ntt.data(),
                            sk.tlwe_key_device_location_.data(),
                            context_->ntt_table_->data(), context_->prime_,
                            bk_N);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        tfhe_generate_bootkey_kernel<<<bk_n, 512,
                                                       sizeof(Data64) * bk_N,
                                                       options.stream_>>>(
                            tlwe_key_ntt.data(),
                            sk.lwe_key_device_location_.data(),
                            temp_boot_key.data(), context_->ntt_table_->data(),
                            context_->intt_table_->data(), context_->n_inverse_,
                            context_->prime_, bk_N, bk_k, bk_base_bit_size,
                            bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.boot_key_device_location_.resize(total_bootkey_size,
                                                            options.stream_);
                        tfhe_convert_bootkey_ntt_domain_kernel<<<
                            bk_n, 512, sizeof(Data64) * bk_N,
                            options.stream_>>>(
                            bk.boot_key_device_location_.data(),
                            temp_boot_key.data(), context_->ntt_table_->data(),
                            context_->prime_, bk_N, bk_k, bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.boot_key_variances_ = std::vector<double>(
                            bk_n * (bk_k + 1) * bk_length,
                            context_->bk_stdev_ * context_->bk_stdev_);

                        // Switch Key
                        int ks_n = context_->n_;
                        int ks_N = context_->k_ * context_->N_;
                        int ks_base_bit_size = bk.ks_base_bit_;
                        int ks_length = bk.ks_length_;
                        int ks_base = 1 << ks_base_bit_size;

                        Data64 total_noise_size = (Data64) ks_N *
                                                  (Data64) ks_length *
                                                  (Data64) (ks_base - 1);
                        DeviceVector<double> noise(total_noise_size,
                                                   options.stream_);

                        detail::PrivateRandomBuffer<double> switch_noise(
                            total_noise_size);
                        std::normal_distribution<double> noise_distribution(
                            0.0, context_->ks_stdev_);
                        for (auto& value : switch_noise.values)
                            value = noise_distribution(random);
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            noise.data(), switch_noise.values.data(),
                            total_noise_size * sizeof(double),
                            cudaMemcpyHostToDevice, options.stream_));
                        Data64 total_random_number_size =
                            total_noise_size * ks_n;
                        detail::PrivateRandomBuffer<int32_t> switch_random(
                            total_random_number_size);
                        detail::private_random_bytes(
                            switch_random.values.data(),
                            total_random_number_size * sizeof(int32_t));
                        bk.switch_key_device_location_a_.resize(
                            total_random_number_size, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            bk.switch_key_device_location_a_.data(),
                            switch_random.values.data(),
                            total_random_number_size * sizeof(int32_t),
                            cudaMemcpyHostToDevice, options.stream_));
                        HEONGPU_CUDA_CHECK(
                            cudaStreamSynchronize(options.stream_));

                        size_t smem = (512 / 32 + 1) * sizeof(uint32_t);

                        bk.switch_key_device_location_b_.resize(
                            total_noise_size, options.stream_);
                        tfhe_generate_switchkey_kernel<<<ks_N, 512, smem,
                                                         options.stream_>>>(
                            sk.tlwe_key_device_location_.data(),
                            sk.lwe_key_device_location_.data(), noise.data(),
                            bk.switch_key_device_location_a_.data(),
                            bk.switch_key_device_location_b_.data(), ks_n,
                            ks_base_bit_size, ks_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.switch_key_variances_ = std::vector<double>(
                            total_noise_size,
                            context_->ks_stdev_ * context_->ks_stdev_);
                        bk.storage_type_ = storage_type::DEVICE;
                        bk.boot_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

} // namespace heongpu
