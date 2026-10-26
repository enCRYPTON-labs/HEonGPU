// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/encoder.cuh>

namespace heongpu
{

    __host__ HEEncoder<Scheme::BFV>::HEEncoder(HEContext<Scheme::BFV> context)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;

        // Encode - Decode Index
        std::vector<Data64> encode_index;

        int m = n << 1;
        int gen = 3;
        int pos = 1;
        int index = 0;
        int location = 0;
        for (int i = 0; i < int(n / 2); i++)
        {
            index = (pos - 1) >> 1;
            location = gpuntt::bitreverse(index, n_power);
            encode_index.push_back(location);
            pos *= gen;
            pos &= (m - 1);
        }
        for (int i = int(n / 2); i < n; i++)
        {
            index = (m - pos - 1) >> 1;
            location = gpuntt::bitreverse(index, n_power);
            encode_index.push_back(location);
            pos *= gen;
            pos &= (m - 1);
        }

        encoding_location_ =
            std::make_shared<DeviceVector<Data64>>(encode_index);
    }

    __host__ void
    HEEncoder<Scheme::BFV>::encode_bfv(Plaintext<Scheme::BFV>& plain,
                                       const std::vector<uint64_t>& message,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> output_memory(n, stream);

        DeviceVector<Data64> message_gpu(slot_count, stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data64), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            encoding_location_->data(), ctx->plain_modulus2_->data(),
            message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_plain_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                 ctx->plain_intt_tables_->data(),
                                 ctx->plain_modulus2_->data(), cfg_intt, 1, 1);

        plain.memory_set(std::move(output_memory));
    }

    __host__ void
    HEEncoder<Scheme::BFV>::encode_bfv(Plaintext<Scheme::BFV>& plain,
                                       const std::vector<int64_t>& message,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> output_memory(n, stream);

        DeviceVector<Data64> message_gpu(slot_count, stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data64), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            encoding_location_->data(), ctx->plain_modulus2_->data(),
            message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_plain_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                 ctx->plain_intt_tables_->data(),
                                 ctx->plain_modulus2_->data(), cfg_intt, 1, 1);

        plain.memory_set(std::move(output_memory));
    }

    __host__ void
    HEEncoder<Scheme::BFV>::encode_bfv(Plaintext<Scheme::BFV>& plain,
                                       const HostVector<uint64_t>& message,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> output_memory(n, stream);

        DeviceVector<Data64> message_gpu(slot_count, stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data64), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            encoding_location_->data(), ctx->plain_modulus2_->data(),
            message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_plain_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                 ctx->plain_intt_tables_->data(),
                                 ctx->plain_modulus2_->data(), cfg_intt, 1, 1);

        plain.memory_set(std::move(output_memory));
    }

    __host__ void
    HEEncoder<Scheme::BFV>::encode_bfv(Plaintext<Scheme::BFV>& plain,
                                       const HostVector<int64_t>& message,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> output_memory(n, stream);

        DeviceVector<Data64> message_gpu(slot_count, stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data64), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            encoding_location_->data(), ctx->plain_modulus2_->data(),
            message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_plain_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                 ctx->plain_intt_tables_->data(),
                                 ctx->plain_modulus2_->data(), cfg_intt, 1, 1);

        plain.memory_set(std::move(output_memory));
    }

    __host__ void
    HEEncoder<Scheme::BFV>::decode_bfv(std::vector<uint64_t>& message,
                                       Plaintext<Scheme::BFV>& plain,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> temp_memory(slot_count + n, stream);
        Data64* message_gpu = temp_memory.data();
        Data64* temp_plain = message_gpu + slot_count;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(plain.data(), temp_plain,
                        ctx->plain_ntt_tables_->data(),
                        ctx->plain_modulus2_->data(), cfg_ntt, 1, 1);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            message_gpu, temp_plain, encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count);

        cudaMemcpyAsync(message.data(), message_gpu,
                        slot_count * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::BFV>::decode_bfv(std::vector<int64_t>& message,
                                       Plaintext<Scheme::BFV>& plain,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> temp_memory(slot_count + n, stream);
        Data64* message_gpu = temp_memory.data();
        Data64* temp_plain = message_gpu + slot_count;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(plain.data(), temp_plain,
                        ctx->plain_ntt_tables_->data(),
                        ctx->plain_modulus2_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            message_gpu, temp_plain, encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count >> 8), 1, 1), 256, 0,
                                  stream>>>(message_gpu, message_gpu,
                                            ctx->plain_modulus2_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count);

        cudaMemcpyAsync(message.data(), message_gpu,
                        slot_count * sizeof(int64_t), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::BFV>::decode_bfv(HostVector<uint64_t>& message,
                                       Plaintext<Scheme::BFV>& plain,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> temp_memory(slot_count + n, stream);
        Data64* message_gpu = temp_memory.data();
        Data64* temp_plain = message_gpu + slot_count;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(plain.data(), temp_plain,
                        ctx->plain_ntt_tables_->data(),
                        ctx->plain_modulus2_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            message_gpu, temp_plain, encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count);

        cudaMemcpyAsync(message.data(), message_gpu,
                        slot_count * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::BFV>::decode_bfv(HostVector<int64_t>& message,
                                       Plaintext<Scheme::BFV>& plain,
                                       const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int slot_count = ctx->n;

        DeviceVector<Data64> temp_memory(slot_count + n, stream);
        Data64* message_gpu = temp_memory.data();
        Data64* temp_plain = message_gpu + slot_count;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(plain.data(), temp_plain,
                        ctx->plain_ntt_tables_->data(),
                        ctx->plain_modulus2_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            message_gpu, temp_plain, encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count >> 8), 1, 1), 256, 0,
                                  stream>>>(message_gpu, message_gpu,
                                            ctx->plain_modulus2_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count);

        cudaMemcpyAsync(message.data(), message_gpu,
                        slot_count * sizeof(int64_t), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
