// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/keygenerator.cuh>
#include "../secure_random.h"

namespace heongpu
{
    __host__ HEKeyGenerator<Scheme::CKKS>::HEKeyGenerator(
        HEContext<Scheme::CKKS> context)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        new_seed_ = RNGSeed();
    }

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_secret_key(
        Secretkey<Scheme::CKKS>& sk, const ExecutionOptions& options)
    {
        if (options.storage_ != storage_type::DEVICE &&
            options.storage_ != storage_type::HOST)
            throw ParameterError(error_message(
                "Invalid storage type ", static_cast<int>(options.storage_),
                "! Expected storage_type::DEVICE or storage_type::HOST."));
        sk.validate_context(context_);
        if (sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is already generated!");
        }

        output_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                DeviceVector<int> secret_key_without_rns((context_->n),
                                                         options.stream_);
                HEONGPU_CUDA_CHECK(cudaMemsetAsync(
                    secret_key_without_rns.data(), 0, context_->n * sizeof(int),
                    options.stream_));

                detail::PrivateRandomBuffer<int> positions(sk_.hamming_weight_);
                detail::PrivateRandomBuffer<int> values(sk_.hamming_weight_);
                detail::sample_ternary_positions(
                    context_->n, sk_.hamming_weight_, positions.values,
                    values.values);
                DeviceVector<int> positions_device(positions.values,
                                                   options.stream_);
                DeviceVector<int> values_device(values.values, options.stream_);
                secretkey_gen_kernel<<<dim3((context_->n >> 8), 1, 1), 256, 0,
                                       options.stream_>>>(
                    secret_key_without_rns.data(), positions_device.data(),
                    values_device.data(), sk_.hamming_weight_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                // Keep private host samples alive until their uploads complete.
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));

                DeviceVector<Data64> secret_key_rns(
                    (sk_.coeff_modulus_count() * context_->n), options.stream_);

                secretkey_rns_kernel<<<dim3((context_->n >> 8), 1, 1), 256, 0,
                                       options.stream_>>>(
                    secret_key_without_rns.data(), secret_key_rns.data(),
                    context_->modulus_->data(), context_->n_power,
                    context_->Q_prime_size);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = context_->n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .ntt_layout = gpuntt::PerPolynomial,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = options.stream_};

                gpuntt::GPU_NTT_Inplace(
                    secret_key_rns.data(), context_->ntt_table_->data(),
                    context_->modulus_->data(), cfg_ntt, context_->Q_prime_size,
                    context_->Q_prime_size);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                sk_.in_ntt_domain_ = true;
                sk_.secret_key_generated_ = true;

                sk_.memory_set(std::move(secret_key_rns));
            },
            options);
    }

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_public_key(
        Publickey<Scheme::CKKS>& pk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw StateError("Secretkey is not generated! Call "
                             "HEKeyGenerator::generate_secret_key first.");
        }

        if (pk.public_key_generated_)
        {
            throw StateError("Publickey is already generated! "
                             "Use a new Publickey object.");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    pk,
                    [&](Publickey<Scheme::CKKS>& pk_)
                    {
                        DeviceVector<Data64> output_memory(
                            (2 * context_->Q_prime_size * context_->n),
                            options.stream_);

                        DeviceVector<Data64> errors_a(
                            2 * context_->Q_prime_size * context_->n,
                            options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly =
                            error_poly + (context_->Q_prime_size * context_->n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size, 1,
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size, 1, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            errors_a.data(), context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->Q_prime_size, context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        publickey_gen_kernel<<<dim3((context_->n >> 8),
                                                    context_->Q_prime_size, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk_.data(), error_poly,
                            a_poly, context_->modulus_->data(),
                            context_->n_power, context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        pk_.memory_set(std::move(output_memory));

                        pk_.in_ntt_domain_ = true;
                        pk_.public_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_relin_key_method_II(
        Relinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw StateError("Secretkey is not generated! Call "
                             "HEKeyGenerator::generate_secret_key first.");
        }

        if (rk.relin_key_generated_)
        {
            throw StateError("Relinkey is already generated! "
                             "Use a new Relinkey object.");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::CKKS>& rk_)
                    {
                        DeviceVector<Data64> errors_a(
                            2 * context_->Q_prime_size *
                                context_->d_leveled->operator[](0) *
                                context_->n,
                            options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly =
                            error_poly +
                            (context_->Q_prime_size *
                             context_->d_leveled->operator[](0) * context_->n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->d_leveled->operator[](0) *
                                context_->Q_prime_size,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        relinkey_gen_II_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            context_->modulus_->data(),
                            context_->factor_->data(),
                            context_->Sk_pair_leveled->operator[](0).data(),
                            context_->n_power, context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            context_->Q_size, context_->P_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_galois_key_method_II(
        Galoiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw StateError("Secretkey is not generated! Call "
                             "HEKeyGenerator::generate_secret_key first.");
        }

        if (gk.galois_key_generated_)
        {
            throw StateError("Galoiskey is already generated! "
                             "Use a new Galoiskey object.");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                DeviceVector<Data64> errors_a(
                    2 * context_->Q_prime_size *
                        context_->d_leveled->operator[](0) * context_->n,
                    options.stream_);
                Data64* error_poly = errors_a.data();
                Data64* a_poly =
                    error_poly +
                    (context_->Q_prime_size *
                     context_->d_leveled->operator[](0) * context_->n);

                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->d_leveled->operator[](0) *
                                context_->Q_prime_size,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois =
                            modInverse(galois.second, 2 * context_->n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            context_->modulus_->data(),
                            context_->factor_->data(), inv_galois,
                            context_->Sk_pair_leveled->operator[](0).data(),
                            context_->n_power, context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            context_->Q_size, context_->P_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois.second] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois.second] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_));
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, context_->modulus_->data(),
                            context_->n_power, context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly,
                            context_->modulus_->data(), context_->n_power,
                            context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = context_->n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly,
                                            context_->ntt_table_->data(),
                                            context_->modulus_->data(), cfg_ntt,
                                            context_->d_leveled->operator[](0) *
                                                context_->Q_prime_size,
                                            context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((context_->n >> 8),
                                                   context_->Q_prime_size, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero,
                        context_->Sk_pair_leveled->operator[](0).data(),
                        context_->n_power, context_->Q_prime_size,
                        context_->d_leveled->operator[](0), context_->Q_size,
                        context_->P_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_));
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->d_leveled->operator[](0) *
                                context_->Q_prime_size,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * context_->n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            context_->modulus_->data(),
                            context_->factor_->data(), inv_galois,
                            context_->Sk_pair_leveled->operator[](0).data(),
                            context_->n_power, context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            context_->Q_size, context_->P_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois_] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois_] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                                gk.host_location_[galois_].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_));
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, context_->modulus_->data(),
                            context_->n_power, context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly,
                            context_->modulus_->data(), context_->n_power,
                            context_->Q_prime_size,
                            context_->d_leveled->operator[](0),
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = context_->n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly,
                                            context_->ntt_table_->data(),
                                            context_->modulus_->data(), cfg_ntt,
                                            context_->d_leveled->operator[](0) *
                                                context_->Q_prime_size,
                                            context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((context_->n >> 8),
                                                   context_->Q_prime_size, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero,
                        context_->Sk_pair_leveled->operator[](0).data(),
                        context_->n_power, context_->Q_prime_size,
                        context_->d_leveled->operator[](0), context_->Q_size,
                        context_->P_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_));
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_switch_key_method_II(
        Switchkey<Scheme::CKKS>& swk, Secretkey<Scheme::CKKS>& new_sk,
        Secretkey<Scheme::CKKS>& old_sk, const ExecutionOptions& options)
    {
        old_sk.validate_context(context_);
        if (!old_sk.secret_key_generated_)
        {
            throw StateError("Secretkey is not generated! Call "
                             "HEKeyGenerator::generate_secret_key first.");
        }

        new_sk.validate_context(context_);
        if (!new_sk.secret_key_generated_)
        {
            throw std::logic_error("Ner Secretkey is not generated!");
        }

        if (swk.switch_key_generated_)
        {
            throw StateError("Switchkey is already generated! "
                             "Use a new Switchkey object.");
        }

        input_storage_manager(
            old_sk,
            [&](Secretkey<Scheme::CKKS>& old_sk_)
            {
                input_storage_manager(
                    new_sk,
                    [&](Secretkey<Scheme::CKKS>& new_sk_)
                    {
                        output_storage_manager(
                            swk,
                            [&](Switchkey<Scheme::CKKS>& swk_)
                            {
                                DeviceVector<Data64> errors_a(
                                    2 * context_->Q_prime_size *
                                        context_->d_leveled->operator[](0) *
                                        context_->n,
                                    options.stream_);
                                Data64* error_poly = errors_a.data();
                                Data64* a_poly =
                                    error_poly +
                                    (context_->Q_prime_size *
                                     context_->d_leveled->operator[](0) *
                                     context_->n);

                                RandomNumberGenerator::instance()
                                    .modular_uniform_random_number_generation(
                                        a_poly, context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size,
                                        context_->d_leveled->operator[](0),
                                        options.stream_);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, error_poly,
                                        context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size,
                                        context_->d_leveled->operator[](0),
                                        options.stream_);

                                gpuntt::ntt_rns_configuration<Data64> cfg_ntt =
                                    {.n_power = context_->n_power,
                                     .ntt_type = gpuntt::FORWARD,
                                     .ntt_layout = gpuntt::PerPolynomial,
                                     .reduction_poly =
                                         gpuntt::ReductionPolynomial::X_N_plus,
                                     .zero_padding = false,
                                     .stream = options.stream_};

                                gpuntt::GPU_NTT_Inplace(
                                    error_poly, context_->ntt_table_->data(),
                                    context_->modulus_->data(), cfg_ntt,
                                    context_->d_leveled->operator[](0) *
                                        context_->Q_prime_size,
                                    context_->Q_prime_size);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    swk_.switchkey_size_, options.stream_);

                                switchkey_gen_II_kernel<<<
                                    dim3((context_->n >> 8),
                                         context_->Q_prime_size, 1),
                                    256, 0, options.stream_>>>(
                                    output_memory.data(), new_sk.data(),
                                    old_sk.data(), error_poly, a_poly,
                                    context_->modulus_->data(),
                                    context_->factor_->data(),
                                    context_->Sk_pair_leveled->operator[](0)
                                        .data(),
                                    context_->n_power, context_->Q_prime_size,
                                    context_->d_leveled->operator[](0),
                                    context_->Q_size, context_->P_size);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                swk.memory_set(std::move(output_memory));

                                swk.switch_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

} // namespace heongpu
