// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/decryptor.cuh>
#include <algorithm>

namespace heongpu
{
    __host__
    HEDecryptor<Scheme::BFV>::HEDecryptor(HEContext<Scheme::BFV> context,
                                          Secretkey<Scheme::BFV>& secret_key)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        context_ = std::move(context);
        secret_key.validate_context(context_);
        if (!secret_key.secret_key_generated_)
            throw std::logic_error("Secretkey is not generated!");

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

    __host__ void
    HEDecryptor<Scheme::BFV>::decrypt_bfv(Plaintext<Scheme::BFV>& plaintext,
                                          Ciphertext<Scheme::BFV>& ciphertext,
                                          const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int Q_size = ctx->Q_size;

        DeviceVector<Data64> output_memory(n, stream);

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size << n_power);

        DeviceVector<Data64> temp_memory(2 * n * Q_size, stream);
        Data64* ct0_temp = temp_memory.data();
        Data64* ct1_temp = temp_memory.data() + (Q_size << n_power);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!ciphertext.in_ntt_domain_)
        {
            gpuntt::GPU_NTT(ct1, ct1_temp, ctx->ntt_table_->data(),
                            ctx->modulus_->data(), cfg_ntt, Q_size, Q_size);

            sk_multiplication<<<dim3((n >> 8), Q_size, 1), 256, 0, stream>>>(
                ct1_temp, secret_key_.data(), ct1_temp, ctx->modulus_->data(),
                n_power, Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((n >> 8), Q_size, 1), 256, 0, stream>>>(
                ct1, secret_key_.data(), ct1_temp, ctx->modulus_->data(),
                n_power, Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_inverse_->data(),
            .stream = stream};

        if (ciphertext.in_ntt_domain_)
        {
            // TODO: merge these NTTs
            gpuntt::GPU_INTT(ct0, ct0_temp, ctx->intt_table_->data(),
                             ctx->modulus_->data(), cfg_intt, Q_size, Q_size);

            gpuntt::GPU_INTT_Inplace(ct1_temp, ctx->intt_table_->data(),
                                     ctx->modulus_->data(), cfg_intt, Q_size,
                                     Q_size);

            ct0 = ct0_temp;
        }
        else
        {
            gpuntt::GPU_INTT_Inplace(ct1_temp, ctx->intt_table_->data(),
                                     ctx->modulus_->data(), cfg_intt, Q_size,
                                     Q_size);
        }

        decryption_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1_temp, output_memory.data(), ctx->modulus_->data(),
            ctx->plain_modulus_, ctx->gamma_, ctx->Qi_t_->data(),
            ctx->Qi_gamma_->data(), ctx->Qi_inverse_->data(), ctx->mulq_inv_t_,
            ctx->mulq_inv_gamma_, ctx->inv_gamma_, n_power, Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEDecryptor<Scheme::BFV>::decryptx3_bfv(Plaintext<Scheme::BFV>& plaintext,
                                            Ciphertext<Scheme::BFV>& ciphertext,
                                            const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int Q_size = ctx->Q_size;

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size << n_power);
        Data64* ct2 = ciphertext.data() + (Q_size << (n_power + 1));

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(ct1, ctx->ntt_table_->data(),
                                ctx->modulus_->data(), cfg_ntt, 2 * Q_size,
                                Q_size);

        sk_multiplicationx3<<<dim3((n >> 8), Q_size, 1), 256, 0, stream>>>(
            ct1, secret_key_.data(), ctx->modulus_->data(), n_power, Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(ct1, ctx->intt_table_->data(),
                                 ctx->modulus_->data(), cfg_intt, 2 * Q_size,
                                 Q_size);

        decryption_kernelx3<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1, ct2, plaintext.data(), ctx->modulus_->data(),
            ctx->plain_modulus_, ctx->gamma_, ctx->Qi_t_->data(),
            ctx->Qi_gamma_->data(), ctx->Qi_inverse_->data(), ctx->mulq_inv_t_,
            ctx->mulq_inv_gamma_, ctx->inv_gamma_, n_power, Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ int HEDecryptor<Scheme::BFV>::noise_budget_calculation(
        Ciphertext<Scheme::BFV>& ciphertext, const ExecutionOptions& options)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int Q_size = ctx->Q_size;

        HostVector<Data64> max_norm_memory(Q_size);

        input_storage_manager(
            ciphertext,
            [&](Ciphertext<Scheme::BFV>& ciphertext_)
            {
                Data64* ct0 = ciphertext.data();
                Data64* ct1 = ciphertext.data() + (Q_size << n_power);

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .ntt_layout = gpuntt::PerPolynomial,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = options.stream_};

                DeviceVector<Data64> temp_memory(n * Q_size, options.stream_);
                gpuntt::GPU_NTT(ct1, temp_memory.data(),
                                ctx->ntt_table_->data(), ctx->modulus_->data(),
                                cfg_ntt, Q_size, Q_size);

                sk_multiplication<<<dim3((n >> 8), Q_size, 1), 256, 0,
                                    options.stream_>>>(
                    temp_memory.data(), secret_key_.data(), temp_memory.data(),
                    ctx->modulus_->data(), n_power, Q_size);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::INVERSE,
                    .ntt_layout = gpuntt::PerPolynomial,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .mod_inverse = ctx->n_inverse_->data(),
                    .stream = options.stream_};

                gpuntt::GPU_INTT_Inplace(
                    temp_memory.data(), ctx->intt_table_->data(),
                    ctx->modulus_->data(), cfg_intt, Q_size, Q_size);

                coeff_multadd<<<dim3((n >> 8), Q_size, 1), 256, 0,
                                options.stream_>>>(
                    ct0, temp_memory.data(), temp_memory.data(),
                    ctx->plain_modulus_, ctx->modulus_->data(), n_power,
                    Q_size);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                compose_kernel<<<dim3((n >> 8), 1, 1), 256, 0,
                                 options.stream_>>>(
                    temp_memory.data(), temp_memory.data(),
                    ctx->modulus_->data(), ctx->Mi_inv_->data(),
                    ctx->Mi_->data(), ctx->decryption_modulus_->data(), Q_size,
                    n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                find_max_norm_kernel<<<1, 512, sizeof(Data64) * 512,
                                       options.stream_>>>(
                    temp_memory.data(), temp_memory.data(),
                    ctx->upper_half_threshold_->data(),
                    ctx->decryption_modulus_->data(), Q_size,
                    n_power); // TODO: merge with above kernel if possible
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    max_norm_memory.data(), temp_memory.data(),
                    Q_size * sizeof(Data64), cudaMemcpyDeviceToHost,
                    options.stream_));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            },
            options, false);

        bool zero_norm = true;
        for (int i = 0; i < Q_size; i++)
        {
            zero_norm = zero_norm && (max_norm_memory[i] == 0);
        }
        if (zero_norm)
        {
            return ctx->total_bit_count_ - 1;
        }

        return std::max(0, ctx->total_bit_count_ -
                               calculate_big_integer_bit_count(
                                   max_norm_memory.data(), Q_size) -
                               1);
    }

} // namespace heongpu
