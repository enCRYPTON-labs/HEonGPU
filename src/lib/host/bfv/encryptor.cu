// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/encryptor.cuh>

namespace heongpu
{
    __host__
    HEEncryptor<Scheme::BFV>::HEEncryptor(HEContext<Scheme::BFV> context,
                                          Publickey<Scheme::BFV>& public_key)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);
        public_key.validate_context(context_);

        if (public_key.storage_type_ == storage_type::device)
        {
            public_key_ = public_key.device_locations_;
        }
        else
        {
            public_key.store_in_device();
            public_key_ = public_key.device_locations_;
        }

        // context_ holds all parameters
    }

    __host__ void
    HEEncryptor<Scheme::BFV>::encrypt_bfv(Ciphertext<Scheme::BFV>& ciphertext,
                                          Plaintext<Scheme::BFV>& plaintext,
                                          const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int Q_prime_size = ctx->Q_prime_size;
        const int Q_size = ctx->Q_size;
        const int P_size = ctx->P_size;

        DeviceVector<Data64> output_memory((2 * n * Q_size), stream);

        DeviceVector<Data64> gpu_space(5 * Q_prime_size * n, stream);
        Data64* u_poly = gpu_space.data();
        Data64* error_poly = u_poly + (Q_prime_size * n);
        Data64* pk_u_poly = error_poly + (2 * Q_prime_size * n);

        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                u_poly, ctx->modulus_->data(), n_power, Q_prime_size, 1,
                stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, ctx->modulus_->data(), n_power,
                Q_prime_size, 2, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(u_poly, ctx->ntt_table_->data(),
                                ctx->modulus_->data(), cfg_ntt, Q_prime_size,
                                Q_prime_size);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size, 2), 256, 0, stream>>>(
            public_key_.data(), u_poly, pk_u_poly, ctx->modulus_->data(),
            n_power, Q_prime_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(pk_u_poly, ctx->intt_table_->data(),
                                 ctx->modulus_->data(), cfg_intt,
                                 2 * Q_prime_size, Q_prime_size);

        enc_div_lastq_bfv_kernel<<<dim3((n >> 8), Q_size, 2), 256, 0, stream>>>(
            pk_u_poly, error_poly, plaintext.data(), output_memory.data(),
            ctx->modulus_->data(), ctx->half_p_->data(), ctx->half_mod_->data(),
            ctx->last_q_modinv_->data(), ctx->plain_modulus_, ctx->Q_mod_t_,
            ctx->upper_threshold_, ctx->coeeff_div_plainmod_->data(), n_power,
            Q_prime_size, Q_size, P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ciphertext.memory_set(std::move(output_memory));
    }

} // namespace heongpu
