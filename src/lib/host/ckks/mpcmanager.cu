// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/mpcmanager.cuh>

namespace heongpu
{
    __host__ HEMultiPartyManager<Scheme::CKKS>::HEMultiPartyManager(
        HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder,
        double& scale)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        context_ = std::move(context);
        slot_count_ = context_->n >> 1;

        scale_ = scale;
        two_pow_64 = encoder.two_pow_64;
        log_slot_count_ = encoder.log_slot_count_;
        fft_length = encoder.fft_length;

        special_fft_roots_table_ = encoder.special_fft_roots_table_;
        special_ifft_roots_table_ = encoder.special_ifft_roots_table_;
        reverse_order = encoder.reverse_order;

        engine_ = std::mt19937{std::random_device{}()};
        distribution_ = std::normal_distribution<double>(0.0, error_std_dev);
        new_seed_ = RNGSeed();
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::generate_public_key_stage1(
        MultipartyPublickey<Scheme::CKKS>& pk, Secretkey<Scheme::CKKS>& sk,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(
            (2 * context_->Q_prime_size * context_->n), stream);

        RNGSeed common_seed = pk.seed();

        DeviceVector<Data64> errors_a(2 * context_->Q_prime_size * context_->n,
                                      stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (context_->Q_prime_size * context_->n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                a_poly, context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, context_->modulus_->data(),
                context_->n_power, context_->Q_prime_size, 1, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(errors_a.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_prime_size, context_->Q_prime_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        publickey_gen_kernel<<<dim3((context_->n >> 8), context_->Q_prime_size,
                                    1),
                               256, 0, stream>>>(
            output_memory.data(), sk.data(), error_poly, a_poly,
            context_->modulus_->data(), context_->n_power,
            context_->Q_prime_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        pk.memory_set(std::move(output_memory));
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::generate_public_key_stage2(
        std::vector<MultipartyPublickey<Scheme::CKKS>>& all_pk,
        Publickey<Scheme::CKKS>& pk, const cudaStream_t stream)
    {
        int participant_count = all_pk.size();

        DeviceVector<Data64> output_memory(
            (2 * context_->Q_prime_size * context_->n), stream);

        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            context_->Q_prime_size, 2),
                                       256, 0, stream>>>(
            all_pk[0].data(), output_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            threshold_pk_addition<<<dim3((context_->n >> 8),
                                         context_->Q_prime_size, 1),
                                    256, 0, stream>>>(
                all_pk[i].data(), output_memory.data(), output_memory.data(),
                context_->modulus_->data(), context_->n_power, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        pk.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::generate_relin_key_method_I_stage_1(
        MultipartyRelinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_)
                    {
                        RNGSeed common_seed = rk.seed();

                        DeviceVector<Data64> random_values(
                            context_->Q_prime_size *
                                ((3 * context_->Q_size) + 1) * context_->n,
                            options.stream_);
                        Data64* e0 = random_values.data();
                        Data64* e1 = e0 + (context_->Q_prime_size *
                                           context_->Q_size * context_->n);
                        Data64* u = e1 + (context_->Q_prime_size *
                                          context_->Q_size * context_->n);
                        Data64* common_a =
                            u + (context_->Q_prime_size * context_->n);

                        RandomNumberGenerator::instance().set(
                            common_seed.key_, common_seed.nonce_,
                            common_seed.personalization_string_,
                            options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                common_a, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                context_->Q_size, options.stream_);

                        RNGSeed gen_seed1;
                        RandomNumberGenerator::instance().set(
                            gen_seed1.key_, gen_seed1.nonce_,
                            gen_seed1.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, e0, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                2 * context_->Q_size, options.stream_);

                        RandomNumberGenerator::instance().set(
                            new_seed_.key_, new_seed_.nonce_,
                            new_seed_.personalization_string_, options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                u, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size, 1,
                                options.stream_);

                        RNGSeed gen_seed2;
                        RandomNumberGenerator::instance().set(
                            gen_seed2.key_, gen_seed2.nonce_,
                            gen_seed2.personalization_string_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            random_values.data(), context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->Q_prime_size *
                                ((2 * context_->Q_size) + 1),
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_piece_method_I_stage_I_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), common_a, u, e0,
                            e1, context_->modulus_->data(),
                            context_->factor_->data(), context_->n_power,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::generate_relin_key_method_I_stage_3(
        MultipartyRelinkey<Scheme::CKKS>& rk_stage_1,
        MultipartyRelinkey<Scheme::CKKS>& rk_stage_2,
        Secretkey<Scheme::CKKS>& sk, const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!rk_stage_1.relin_key_generated_)
        {
            throw std::logic_error("Relinkey1 is not generated!");
        }

        if (rk_stage_2.relin_key_generated_)
        {
            throw std::logic_error("Relinkey2 is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::CKKS>& rk_stage_2_)
                            {
                                DeviceVector<Data64> random_values(
                                    context_->Q_prime_size *
                                        ((2 * context_->Q_size) + 1) *
                                        context_->n,
                                    options.stream_);
                                Data64* e0 = random_values.data();
                                Data64* e1 =
                                    e0 + (context_->Q_prime_size *
                                          context_->Q_size * context_->n);
                                Data64* u =
                                    e1 + (context_->Q_prime_size *
                                          context_->Q_size * context_->n);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, e0,
                                        context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size, 2,
                                        options.stream_);

                                RandomNumberGenerator::instance().set(
                                    new_seed_.key_, new_seed_.nonce_,
                                    new_seed_.personalization_string_,
                                    options.stream_);
                                RandomNumberGenerator::instance()
                                    .modular_ternary_random_number_generation(
                                        u, context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size, 1,
                                        options.stream_);

                                RNGSeed gen_seed;
                                RandomNumberGenerator::instance().set(
                                    gen_seed.key_, gen_seed.nonce_,
                                    gen_seed.personalization_string_,
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
                                    random_values.data(),
                                    context_->ntt_table_->data(),
                                    context_->modulus_->data(), cfg_ntt,
                                    context_->Q_prime_size *
                                        ((2 * context_->Q_size) + 1),
                                    context_->Q_prime_size);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    rk_stage_2_.relinkey_size_,
                                    options.stream_);

                                multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                                    dim3((context_->n >> 8),
                                         context_->Q_prime_size, 1),
                                    256, 0, options.stream_>>>(
                                    rk_stage_1_.data(), output_memory.data(),
                                    sk.data(), u, e0, e1,
                                    context_->modulus_->data(),
                                    context_->n_power, context_->Q_prime_size,
                                    context_->Q_size);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                rk_stage_2_.memory_set(
                                    std::move(output_memory));

                                rk_stage_2_.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::
        generate_ckks_relin_key_method_II_stage_1(
            MultipartyRelinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
            const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_)
                    {
                        RNGSeed common_seed = rk.seed();

                        DeviceVector<Data64> random_values(
                            context_->Q_prime_size *
                                ((3 * context_->d_leveled->operator[](0)) + 1) *
                                context_->n,
                            options.stream_);
                        Data64* e0 = random_values.data();
                        Data64* e1 = e0 + (context_->Q_prime_size *
                                           context_->d_leveled->operator[](0) *
                                           context_->n);
                        Data64* u = e1 + (context_->Q_prime_size *
                                          context_->d_leveled->operator[](0) *
                                          context_->n);
                        Data64* common_a =
                            u + (context_->Q_prime_size * context_->n);

                        RandomNumberGenerator::instance().set(
                            common_seed.key_, common_seed.nonce_,
                            common_seed.personalization_string_,
                            options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                common_a, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                context_->d_leveled->operator[](0),
                                options.stream_);

                        RNGSeed gen_seed1;
                        RandomNumberGenerator::instance().set(
                            gen_seed1.key_, gen_seed1.nonce_,
                            gen_seed1.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, e0, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                2 * context_->d_leveled->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance().set(
                            new_seed_.key_, new_seed_.nonce_,
                            new_seed_.personalization_string_, options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                u, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size, 1,
                                options.stream_);

                        RNGSeed gen_seed2;
                        RandomNumberGenerator::instance().set(
                            gen_seed2.key_, gen_seed2.nonce_,
                            gen_seed2.personalization_string_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = context_->n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .ntt_layout = gpuntt::PerPolynomial,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            random_values.data(), context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->Q_prime_size *
                                ((2 * context_->d_leveled->operator[](0)) + 1),
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_piece_method_II_stage_I_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), common_a, u, e0,
                            e1, context_->modulus_->data(),
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

    __host__ void HEMultiPartyManager<Scheme::CKKS>::
        generate_ckks_relin_key_method_II_stage_3(
            MultipartyRelinkey<Scheme::CKKS>& rk_stage_1,
            MultipartyRelinkey<Scheme::CKKS>& rk_stage_2,
            Secretkey<Scheme::CKKS>& sk, const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!rk_stage_1.relin_key_generated_)
        {
            throw std::logic_error("Relinkey1 is not generated!");
        }

        if (rk_stage_2.relin_key_generated_)
        {
            throw std::logic_error("Relinkey2 is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::CKKS>& rk_stage_2_)
                            {
                                DeviceVector<Data64> random_values(
                                    context_->Q_prime_size *
                                        ((2 *
                                          context_->d_leveled->operator[](0)) +
                                         1) *
                                        context_->n,
                                    options.stream_);
                                Data64* e0 = random_values.data();
                                Data64* e1 =
                                    e0 + (context_->Q_prime_size *
                                          context_->d_leveled->operator[](0) *
                                          context_->n);
                                Data64* u =
                                    e1 + (context_->Q_prime_size *
                                          context_->d_leveled->operator[](0) *
                                          context_->n);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, e0,
                                        context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size, 2,
                                        options.stream_);

                                RandomNumberGenerator::instance().set(
                                    new_seed_.key_, new_seed_.nonce_,
                                    new_seed_.personalization_string_,
                                    options.stream_);
                                RandomNumberGenerator::instance()
                                    .modular_ternary_random_number_generation(
                                        u, context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size, 1,
                                        options.stream_);

                                RNGSeed gen_seed;
                                RandomNumberGenerator::instance().set(
                                    gen_seed.key_, gen_seed.nonce_,
                                    gen_seed.personalization_string_,
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
                                    random_values.data(),
                                    context_->ntt_table_->data(),
                                    context_->modulus_->data(), cfg_ntt,
                                    context_->Q_prime_size *
                                        ((2 *
                                          context_->d_leveled->operator[](0)) +
                                         1),
                                    context_->Q_prime_size);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    rk_stage_2.relinkey_size_, options.stream_);

                                multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                                    dim3((context_->n >> 8),
                                         context_->Q_prime_size, 1),
                                    256, 0, options.stream_>>>(
                                    rk_stage_1.data(), output_memory.data(),
                                    sk.data(), u, e0, e1,
                                    context_->modulus_->data(),
                                    context_->n_power, context_->Q_prime_size,
                                    context_->d_leveled->operator[](0));
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                rk_stage_2.memory_set(std::move(output_memory));

                                rk_stage_2.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::generate_relin_key_stage_2(
        std::vector<MultipartyRelinkey<Scheme::CKKS>>& all_rk,
        MultipartyRelinkey<Scheme::CKKS>& rk, const ExecutionOptions& options)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_rk[i].relin_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey is not generated!");
            }
        }

        int dimension;
        switch (static_cast<int>(rk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = rk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = rk.d_;
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }

        input_vector_storage_manager(
            all_rk,
            [&](std::vector<MultipartyRelinkey<Scheme::CKKS>>& all_rk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_)
                    {
                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_method_I_stage_I_kernel<<<
                            dim3((context_->n >> 8), context_->Q_prime_size, 1),
                            256, 0, options.stream_>>>(
                            all_rk[0].data(), output_memory.data(),
                            context_->modulus_->data(), context_->n_power,
                            context_->Q_prime_size, dimension, true);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1; i < participant_count; i++)
                        {
                            multi_party_relinkey_method_I_stage_I_kernel<<<
                                dim3((context_->n >> 8), context_->Q_prime_size,
                                     1),
                                256, 0, options.stream_>>>(
                                all_rk[i].data(), output_memory.data(),
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size, dimension, false);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::generate_relin_key_stage_4(
        std::vector<MultipartyRelinkey<Scheme::CKKS>>& all_rk,
        MultipartyRelinkey<Scheme::CKKS>& rk_common_stage1,
        Relinkey<Scheme::CKKS>& rk, const ExecutionOptions& options)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_rk[i].relin_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey is not generated!");
            }
        }

        if (!rk_common_stage1.relin_key_generated_)
        {
            throw std::logic_error("Common Relinkey is not generated!");
        }

        int dimension;
        switch (static_cast<int>(rk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = rk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = rk.d_;
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }

        input_vector_storage_manager(
            all_rk,
            [&](std::vector<MultipartyRelinkey<Scheme::CKKS>>& all_rk_)
            {
                input_storage_manager(
                    rk_common_stage1,
                    [&](MultipartyRelinkey<Scheme::CKKS>& rk_common_stage1_)
                    {
                        output_storage_manager(
                            rk,
                            [&](Relinkey<Scheme::CKKS>& rk_)
                            {
                                DeviceVector<Data64> output_memory(
                                    rk_.relinkey_size_, options.stream_);

                                multi_party_relinkey_method_I_stage_II_kernel<<<
                                    dim3((context_->n >> 8),
                                         context_->Q_prime_size, 1),
                                    256, 0, options.stream_>>>(
                                    all_rk[0].data(), rk_common_stage1.data(),
                                    output_memory.data(),
                                    context_->modulus_->data(),
                                    context_->n_power, context_->Q_prime_size,
                                    dimension);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                for (int i = 1; i < participant_count; i++)
                                {
                                    multi_party_relinkey_method_I_stage_II_kernel<<<
                                        dim3((context_->n >> 8),
                                             context_->Q_prime_size, 1),
                                        256, 0, options.stream_>>>(
                                        all_rk[i].data(), output_memory.data(),
                                        context_->modulus_->data(),
                                        context_->n_power,
                                        context_->Q_prime_size, dimension);
                                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                                }

                                rk_.memory_set(std::move(output_memory));

                                rk_.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::generate_galois_key_method_I_stage_1(
        MultipartyGaloiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        RNGSeed common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * context_->Q_prime_size *
                                          context_->Q_size * context_->n,
                                      options.stream_);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (context_->Q_prime_size *
                                       context_->Q_size * context_->n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, options.stream_);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                a_poly, context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, context_->Q_size, options.stream_);

        RNGSeed gen_seed1;
        RandomNumberGenerator::instance().set(gen_seed1.key_, gen_seed1.nonce_,
                                              gen_seed1.personalization_string_,
                                              options.stream_);

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size, context_->Q_size,
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
                            context_->Q_size * context_->Q_prime_size,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois =
                            modInverse(galois.second, 2 * context_->n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((context_->n >> 8),
                                                    context_->Q_prime_size, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            context_->modulus_->data(),
                            context_->factor_->data(), inv_galois,
                            context_->n_power, context_->Q_prime_size);
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
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly,
                            context_->modulus_->data(), context_->n_power,
                            context_->Q_prime_size, context_->Q_size,
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = context_->n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, context_->ntt_table_->data(),
                        context_->modulus_->data(), cfg_ntt,
                        context_->Q_size * context_->Q_prime_size,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((context_->n >> 8),
                                                context_->Q_prime_size, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero, context_->n_power,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly,
                                context_->modulus_->data(), context_->n_power,
                                context_->Q_prime_size, context_->Q_size,
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
                            context_->Q_size * context_->Q_prime_size,
                            context_->Q_prime_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * context_->n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((context_->n >> 8),
                                                    context_->Q_prime_size, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            context_->modulus_->data(),
                            context_->factor_->data(), inv_galois,
                            context_->n_power, context_->Q_prime_size);
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
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly,
                            context_->modulus_->data(), context_->n_power,
                            context_->Q_prime_size, context_->Q_size,
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = context_->n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, context_->ntt_table_->data(),
                        context_->modulus_->data(), cfg_ntt,
                        context_->Q_size * context_->Q_prime_size,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((context_->n >> 8),
                                                context_->Q_prime_size, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero, context_->n_power,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::generate_galois_key_method_II_stage_1(
        MultipartyGaloiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
        const ExecutionOptions& options)
    {
        sk.validate_context(context_);
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        RNGSeed common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * context_->Q_prime_size *
                                          context_->d_leveled->operator[](0) *
                                          context_->n,
                                      options.stream_);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (context_->Q_prime_size *
                          context_->d_leveled->operator[](0) * context_->n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, options.stream_);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                a_poly, context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, context_->d_leveled->operator[](0),
                options.stream_);

        RNGSeed gen_seed1;
        RandomNumberGenerator::instance().set(gen_seed1.key_, gen_seed1.nonce_,
                                              gen_seed1.personalization_string_,
                                              options.stream_);

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
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
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
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
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
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
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
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
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::generate_galois_key_stage_2(
        std::vector<MultipartyGaloiskey<Scheme::CKKS>>& all_gk,
        Galoiskey<Scheme::CKKS>& gk, const ExecutionOptions& options)
    {
        int participant_count = all_gk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common galois!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if ((gk.customized != all_gk[i].customized) ||
                (gk.group_order_ != all_gk[i].group_order_))
            {
                throw std::invalid_argument(
                    "MultipartyGaloiskey context is not valid || "
                    "MultipartyGaloiskey is not generated!");
            }
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_gk[i].galois_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyGaloiskey is not generated!");
            }
        }

        std::vector<storage_type> input_storage_types;
        for (int i = 0; i < participant_count; i++)
        {
            input_storage_types.push_back(all_gk[i].storage_type_);

            if (all_gk[i].storage_type_ == storage_type::DEVICE)
            {
                for (const auto& pair : all_gk[i].device_location_)
                {
                    if (pair.second.size() < all_gk[i].galoiskey_size_)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys size is not valid || "
                            "MultipartyGaloiskeys is not generated!");
                    }
                }
            }
            else
            {
                for (const auto& pair : all_gk[i].host_location_)
                {
                    if (pair.second.size() < all_gk[i].galoiskey_size_)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys size is not valid || "
                            "MultipartyGaloiskeys is not generated!");
                    }
                }

                all_gk[i].store_in_device();
            }
        }

        int dimension;
        switch (static_cast<int>(gk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = gk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = gk.d_;
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }

        for (auto& galois : gk.galois_elt)
        {
            DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                               options.stream_);

            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                options.stream_>>>(
                output_memory.data(),
                all_gk[0].device_location_[galois.second].data(),
                context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, dimension, true);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            for (int i = 1; i < participant_count; i++)
            {
                multi_party_galoiskey_gen_method_I_II_kernel<<<
                    dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                    options.stream_>>>(
                    output_memory.data(),
                    all_gk[i].device_location_[galois.second].data(),
                    context_->modulus_->data(), context_->n_power,
                    context_->Q_prime_size, dimension, false);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            if (options.storage_ == storage_type::DEVICE)
            {
                gk.device_location_[galois.second] = std::move(output_memory);
            }
            else
            {
                gk.host_location_[galois.second] =
                    HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }

        DeviceVector<Data64> output_memory(gk.galoiskey_size_, options.stream_);

        multi_party_galoiskey_gen_method_I_II_kernel<<<
            dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
            options.stream_>>>(output_memory.data(),
                               all_gk[0].zero_device_location_.data(),
                               context_->modulus_->data(), context_->n_power,
                               context_->Q_prime_size, dimension, true);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                options.stream_>>>(
                output_memory.data(), all_gk[i].zero_device_location_.data(),
                context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, dimension, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (options.storage_ == storage_type::DEVICE)
        {
            gk.zero_device_location_ = std::move(output_memory);
        }
        else
        {
            gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
            cudaMemcpyAsync(gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (options.keep_initial_condition_)
        {
            for (int i = 0; i < participant_count; i++)
            {
                if (input_storage_types[i] == storage_type::DEVICE)
                {
                    // pass
                }
                else
                {
                    all_gk[i].store_in_host();
                }
            }
        }

        gk.storage_type_ = options.storage_;
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::partial_decrypt_stage_1(
        Ciphertext<Scheme::CKKS>& ciphertext, Secretkey<Scheme::CKKS>& sk,
        Ciphertext<Scheme::CKKS>& partial_ciphertext, const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - ciphertext.depth_;

        Data64* ct0 = ciphertext.data();
        Data64* ct1 =
            ciphertext.data() + (current_decomp_count << context_->n_power);

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        sk_multiplication<<<dim3((context_->n >> 8), current_decomp_count, 1),
                            256, 0, stream>>>(
            ct1, sk.data(),
            output_memory.data() + (current_decomp_count << context_->n_power),
            context_->modulus_->data(), context_->n_power, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Data64> error_poly(current_decomp_count * context_->n,
                                        stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly.data(), context_->modulus_->data(),
                context_->n_power, current_decomp_count, 1, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                current_decomp_count, current_decomp_count);

        // TODO: Optimize it!
        addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256, 0,
                   stream>>>(
            output_memory.data() + (current_decomp_count * context_->n),
            error_poly.data(),
            output_memory.data() + (current_decomp_count * context_->n),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            current_decomp_count, 1),
                                       256, 0, stream>>>(
            ct0, output_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        partial_ciphertext.memory_set(std::move(output_memory));
    }

    __host__ void HEMultiPartyManager<Scheme::CKKS>::partial_decrypt_stage_2(
        std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
        Plaintext<Scheme::CKKS>& plaintext, const cudaStream_t stream)
    {
        int cipher_count = ciphertexts.size();
        int current_detph = ciphertexts[0].depth_;
        int current_decomp_count = context_->Q_size - current_detph;

        DeviceVector<Data64> output_memory(context_->n * current_decomp_count,
                                           stream);

        Data64* ct0 = ciphertexts[0].data();
        Data64* ct1 =
            ciphertexts[0].data() + (current_decomp_count << context_->n_power);
        addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256, 0,
                   stream>>>(ct0, ct1, output_memory.data(),
                             context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            Data64* ct1_i = ciphertexts[i].data() +
                            (current_decomp_count << context_->n_power);

            addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256,
                       0, stream>>>(
                ct1_i, output_memory.data(), output_memory.data(),
                context_->modulus_->data(), context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::distributed_bootstrapping_stage1(
        Ciphertext<Scheme::CKKS>& common, Ciphertext<Scheme::CKKS>& output,
        Secretkey<Scheme::CKKS>& secret_key, const RNGSeed& seed,
        const cudaStream_t stream)
    {
        // Random message generation on CPU
        std::vector<double> random_message;
        for (std::size_t i = 0; i < slot_count_; ++i)
        {
            random_message.push_back(distribution_(engine_));
        }

        // ------------------ Encoding Stage ------------------
        DeviceVector<double> random_message_gpu(slot_count_, stream);
        cudaMemcpyAsync(random_message_gpu.data(), random_message.data(),
                        random_message.size() * sizeof(double),
                        cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Complex64> temp_complex(context_->n, stream);
        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(random_message_gpu.data(),
                                             temp_complex.data());

        double fix = scale_ / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        DeviceVector<Data64> random_message_rns_gpu(
            context_->n * context_->Q_size, stream);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            random_message_rns_gpu.data(), temp_complex.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(random_message_rns_gpu.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        // ------------------ Random Decryption Stage ------------------
        int current_decomp_count = context_->Q_size - common.depth_;
        RNGSeed common_seed = seed;

        // TODO: Optimize it!
        DeviceVector<Data64> temp_vector(
            ((2 * context_->Q_size) + current_decomp_count) * context_->n,
            stream);
        Data64* error_poly0 = temp_vector.data();
        Data64* error_poly1 =
            error_poly0 + (current_decomp_count * context_->n);
        Data64* a_poly = error_poly1 + (context_->Q_size * context_->n);

        // a poly generation (assume in NTT domain)
        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                a_poly, context_->modulus_->data(), context_->n_power,
                context_->Q_size, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        // error0 poly generation
        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly0, context_->modulus_->data(),
                context_->n_power, current_decomp_count, 1, stream);

        gpuntt::GPU_NTT_Inplace(error_poly0, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                current_decomp_count, current_decomp_count);

        // error1 poly generation
        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly1, context_->modulus_->data(),
                context_->n_power, context_->Q_size, 1, stream);

        gpuntt::GPU_NTT_Inplace(error_poly1, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        DeviceVector<Data64> output_memory(
            ((current_decomp_count + context_->Q_size) * context_->n), stream);

        Data64* ct0 = common.data();
        Data64* ct1 =
            common.data() + (current_decomp_count << context_->n_power);

        if (!common.in_ntt_domain_)
        {
            DeviceVector<Data64> temp_memory(context_->n * current_decomp_count,
                                             stream);
            gpuntt::GPU_NTT(ct1, temp_memory.data(),
                            context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            current_decomp_count, current_decomp_count);

            col_boot_dec_mul_with_sk_ckks<<<
                dim3((context_->n >> 8),
                     (current_decomp_count + context_->Q_size), 1),
                256, 0, stream>>>(temp_memory.data(), a_poly, secret_key.data(),
                                  output_memory.data(),
                                  context_->modulus_->data(), context_->n_power,
                                  context_->Q_size, current_decomp_count);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            col_boot_dec_mul_with_sk_ckks<<<
                dim3((context_->n >> 8),
                     (current_decomp_count + context_->Q_size), 1),
                256, 0, stream>>>(ct1, a_poly, secret_key.data(),
                                  output_memory.data(),
                                  context_->modulus_->data(), context_->n_power,
                                  context_->Q_size, current_decomp_count);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        col_boot_add_random_and_errors_ckks<<<
            dim3((context_->n >> 8), (current_decomp_count + context_->Q_size),
                 1),
            256, 0, stream>>>(output_memory.data(), error_poly0, error_poly1,
                              random_message_rns_gpu.data(),
                              context_->modulus_->data(), context_->n_power,
                              context_->Q_size, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::CKKS>::distributed_bootstrapping_stage2(
        std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
        Ciphertext<Scheme::CKKS>& common, Ciphertext<Scheme::CKKS>& output,
        const RNGSeed& seed, const cudaStream_t stream)
    {
        RNGSeed common_seed = seed;
        int cipher_count = ciphertexts.size();
        int current_decomp_count = context_->Q_size - common.depth_;

        // ------------------ h0, h1 Accumulation Stage ------------------

        // TODO: Make efficient!
        DeviceVector<Data64> h_memory(
            ((current_decomp_count + context_->Q_size) * context_->n), stream);
        Data64* h0 = h_memory.data();
        Data64* h1 =
            h_memory.data() + (current_decomp_count << context_->n_power);

        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            current_decomp_count, 1),
                                       256, 0, stream>>>(ciphertexts[0].data(),
                                                         h0, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 1), 256, 0, stream>>>(
            ciphertexts[0].data() + ((current_decomp_count) *context_->n), h1,
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256,
                       0, stream>>>(ciphertexts[i].data(), h0, h0,
                                    context_->modulus_->data(),
                                    context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            addition<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                       stream>>>(
                ciphertexts[i].data() + ((current_decomp_count) *context_->n),
                h1, h1, context_->modulus_->data(), context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        DeviceVector<Data64> rand_message_memory(
            context_->n * current_decomp_count, stream);

        addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256, 0,
                   stream>>>(h0, common.data(), rand_message_memory.data(),
                             context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // ------------------ Decoding Stage ------------------

        DeviceVector<double> message_gpu(slot_count_, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(rand_message_memory.data(),
                                 context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 current_decomp_count, current_decomp_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < common.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        DeviceVector<Complex64> temp_complex(context_->n, stream);
        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream>>>(
            temp_complex.data(), rand_message_memory.data(),
            context_->modulus_->data(), context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_decomp_count, common.scale_, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpufft::fft_configuration<Float64> cfg_fft{};
        cfg_fft.n_power = log_slot_count_;
        cfg_fft.fft_type = gpufft::type::FORWARD;
        cfg_fft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(temp_complex.data(),
                                             message_gpu.data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // ------------------ Encoding Stage ------------------

        DeviceVector<Data64> output_memory(2 * context_->n * context_->Q_size,
                                           stream);

        Data64* o_c0 = output_memory.data();
        Data64* o_c1 =
            output_memory.data() + (context_->Q_size << context_->n_power);

        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(message_gpu.data(),
                                             temp_complex.data());

        double fix = scale_ / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            o_c0, temp_complex.data(), context_->modulus_->data(),
            context_->Q_size, two_pow_64, reverse_order->data(),
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(o_c0, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        // ------------------ Fresh Cipher Stage ------------------

        addition<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                   stream>>>(o_c0, h1, o_c0, context_->modulus_->data(),
                             context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                o_c1, context_->modulus_->data(), context_->n_power,
                context_->Q_size, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        output.memory_set(std::move(output_memory));
    }
} // namespace heongpu
