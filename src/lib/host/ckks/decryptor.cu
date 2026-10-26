// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/decryptor.cuh>

namespace heongpu
{

    __host__
    HEDecryptor<Scheme::CKKS>::HEDecryptor(HEContext<Scheme::CKKS> context,
                                           Secretkey<Scheme::CKKS>& secret_key)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);
        secret_key.validate_context(context_);
        if (!secret_key.secret_key_generated_)
            throw StateError("Secretkey is not generated! Call "
                             "HEKeyGenerator::generate_secret_key first.");

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        if (secret_key.storage_type_ == storage_type::DEVICE)
        {
            secret_key_ = secret_key.device_locations_;
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(secret_key_.stream()));
        }
        else
        {
            secret_key.store_in_device();
            secret_key_ = secret_key.device_locations_;
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(secret_key_.stream()));
        }
    }

    __host__ void HEDecryptor<Scheme::CKKS>::decrypt_ckks(
        Plaintext<Scheme::CKKS>& plaintext,
        Ciphertext<Scheme::CKKS>& ciphertext, const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int current_decomp_count = ctx->Q_size - ciphertext.depth_;
        DeviceVector<Data64> output_memory(n * current_decomp_count, stream);

        sk_multiplication_ckks<<<dim3((n >> 8), current_decomp_count, 1), 256,
                                 0, stream>>>(
            ciphertext.data(), output_memory.data(), secret_key_.data(),
            ctx->modulus_->data(), n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

} // namespace heongpu
