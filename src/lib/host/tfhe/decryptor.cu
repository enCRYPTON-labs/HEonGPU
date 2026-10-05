// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/decryptor.cuh>

namespace heongpu
{
    __host__
    HEDecryptor<Scheme::TFHE>::HEDecryptor(HEContext<Scheme::TFHE> context,
                                           Secretkey<Scheme::TFHE>& secret_key)
    {
        if (!context)
        {
            throw std::invalid_argument("HEContext is not set!");
        }

        context_ = std::move(context);

        secret_key.validate_context(context_);
        if (!secret_key.secret_key_generated_)
            throw std::logic_error("Secretkey is not generated!");
        if (secret_key.is_on_device())
            lwe_key_device_location_ = secret_key.lwe_key_device_location_;
        else
            lwe_key_device_location_ =
                DeviceVector<int32_t>(secret_key.lwe_key_host_location_);
        HEONGPU_CUDA_CHECK(
            cudaStreamSynchronize(lwe_key_device_location_.stream()));
    }

    __host__ void
    HEDecryptor<Scheme::TFHE>::decrypt_lwe(std::vector<bool>& messages,
                                           Ciphertext<Scheme::TFHE>& ciphertext,
                                           const cudaStream_t stream)
    {
        const int THREADS = 512;
        int block_count = ciphertext.shape_;
        size_t smem = (THREADS / 32 + 1) * sizeof(uint32_t);

        DeviceVector<int32_t> messages_gpu(ciphertext.shape_, stream);

        if (block_count == 0)
        {
            messages.clear();
            return;
        }
        decrypt_lwe_kernel<<<block_count, THREADS, smem, stream>>>(
            lwe_key_device_location_.data(),
            ciphertext.a_device_location_.data(),
            ciphertext.b_device_location_.data(), messages_gpu.data(),
            context_->n_, ciphertext.shape_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        std::vector<int32_t> messages_encoded(ciphertext.shape_);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(messages_encoded.data(),
                                           messages_gpu.data(),
                                           ciphertext.shape_ * sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        messages.resize(ciphertext.shape_);
        for (int i = 0; i < ciphertext.shape_; i++)
        {
            messages[i] = (messages_encoded[i] > 0);
        }
    }

} // namespace heongpu
