// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/encryptor.cuh>
#include "../secure_random.h"

namespace heongpu
{
    __host__
    HEEncryptor<Scheme::TFHE>::HEEncryptor(HEContext<Scheme::TFHE> context,
                                           Secretkey<Scheme::TFHE>& secret_key)
    {
        if (!context)
        {
            throw std::invalid_argument("HEContext is not set!");
        }

        context_ = std::move(context);

        secret_key.validate_context(context_);
        if (!secret_key.secret_key_generated_)
        {
            throw std::runtime_error("Secretkey was not generated!");
        }

        alpha_min_ = secret_key.lwe_alpha_min;
        alpha_max_ = secret_key.lwe_alpha_max;

        if (secret_key.storage_type_ == storage_type::DEVICE)
        {
            lwe_key_device_location_ = secret_key.lwe_key_device_location_;
        }
        else
        {
            lwe_key_device_location_ =
                DeviceVector<int32_t>(secret_key.lwe_key_host_location_);
        }

        HEONGPU_CUDA_CHECK(
            cudaStreamSynchronize(lwe_key_device_location_.stream()));
    }

    __host__ void HEEncryptor<Scheme::TFHE>::encrypt_lwe_symmetric(
        Ciphertext<Scheme::TFHE>& ciphertext, std::vector<int32_t>& messages,
        const cudaStream_t stream)
    {
        detail::SecretKeyRandomGenerator random;
        std::normal_distribution<double> dist(0.0, alpha_min_);
        detail::PrivateRandomBuffer<int32_t> a(
            static_cast<size_t>(context_->n_) * ciphertext.shape_);
        detail::PrivateRandomBuffer<int32_t> b(ciphertext.shape_);
        detail::private_random_bytes(a.values.data(),
                                     a.values.size() * sizeof(int32_t));
        for (size_t i = 0; i < ciphertext.shape_; ++i)
            b.values[i] =
                static_cast<int32_t>(uint32_t(messages[i]) +
                                     uint32_t(detail::torus32(dist(random))));
        DeviceVector<int32_t> a_device(a.values, stream);
        DeviceVector<int32_t> b_device(b.values, stream);
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        ciphertext.a_device_location_ = std::move(a_device);
        ciphertext.b_device_location_ = std::move(b_device);
        if (ciphertext.shape_ != 0)
        {
            const int threads = 512;
            const int block_count = std::min<size_t>(32, ciphertext.shape_);
            const size_t smem = (threads / 32 + 1) * sizeof(uint32_t);
            encrypt_lwe_kernel<<<block_count, threads, smem, stream>>>(
                lwe_key_device_location_.data(),
                ciphertext.a_device_location_.data(),
                ciphertext.b_device_location_.data(), context_->n_,
                ciphertext.shape_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ciphertext.variances_ =
            std::vector<double>(ciphertext.shape_, alpha_min_ * alpha_min_);
        ciphertext.storage_type_ = storage_type::DEVICE;
    }

    __host__ int32_t
    HEEncryptor<Scheme::TFHE>::encode_to_torus32(uint32_t mu, uint32_t m_size)
    {
        uint64_t interval = ((1ULL << 63) / m_size) * 2;
        uint64_t phase64 = mu * interval;
        return static_cast<int32_t>(phase64 >> 32);
    }

} // namespace heongpu
