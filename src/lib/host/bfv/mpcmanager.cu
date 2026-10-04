// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/mpcmanager.cuh>

namespace heongpu
{
    __host__ HEMultiPartyManager<Scheme::BFV>::HEMultiPartyManager(
        HEContext<Scheme::BFV> context)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }
        context_ = std::move(context);
        new_seed_ = RNGSeed();
    }

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_public_key_stage1(
        MultipartyPublickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_public_key_stage2(
        std::vector<MultipartyPublickey<Scheme::BFV>>& all_pk,
        Publickey<Scheme::BFV>& pk, const cudaStream_t stream)
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

    //

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_relin_key_method_I_stage_1(
        MultipartyRelinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
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
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
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
    HEMultiPartyManager<Scheme::BFV>::generate_relin_key_method_I_stage_3(
        MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
        MultipartyRelinkey<Scheme::BFV>& rk_stage_2, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
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
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_2_)
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

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_bfv_relin_key_method_II_stage_1(
        MultipartyRelinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
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
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
                    {
                        RNGSeed common_seed = rk.seed();

                        DeviceVector<Data64> random_values(
                            context_->Q_prime_size * ((3 * context_->d) + 1) *
                                context_->n,
                            options.stream_);
                        Data64* e0 = random_values.data();
                        Data64* e1 = e0 + (context_->Q_prime_size *
                                           context_->d * context_->n);
                        Data64* u = e1 + (context_->Q_prime_size * context_->d *
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
                                context_->d, options.stream_);

                        RNGSeed gen_seed1;
                        RandomNumberGenerator::instance().set(
                            gen_seed1.key_, gen_seed1.nonce_,
                            gen_seed1.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, e0, context_->modulus_->data(),
                                context_->n_power, context_->Q_prime_size,
                                2 * context_->d, options.stream_);

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
                            context_->Q_prime_size * ((2 * context_->d) + 1),
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
                            context_->Sk_pair_->data(), context_->n_power,
                            context_->Q_prime_size, context_->d,
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
    HEMultiPartyManager<Scheme::BFV>::generate_bfv_relin_key_method_II_stage_3(
        MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
        MultipartyRelinkey<Scheme::BFV>& rk_stage_2, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
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
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_2_)
                            {
                                DeviceVector<Data64> random_values(
                                    context_->Q_prime_size *
                                        ((2 * context_->d) + 1) * context_->n,
                                    options.stream_);
                                Data64* e0 = random_values.data();
                                Data64* e1 = e0 + (context_->Q_prime_size *
                                                   context_->d * context_->n);
                                Data64* u = e1 + (context_->Q_prime_size *
                                                  context_->d * context_->n);

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
                                        ((2 * context_->d) + 1),
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
                                    context_->d);
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_relin_key_stage_2(
        std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk,
        MultipartyRelinkey<Scheme::BFV>& rk, const ExecutionOptions& options)
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
            [&](std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_relin_key_stage_4(
        std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk,
        MultipartyRelinkey<Scheme::BFV>& rk_common_stage1,
        Relinkey<Scheme::BFV>& rk, const ExecutionOptions& options)
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
            [&](std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk_)
            {
                input_storage_manager(
                    rk_common_stage1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_common_stage1_)
                    {
                        output_storage_manager(
                            rk,
                            [&](Relinkey<Scheme::BFV>& rk_)
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

    //

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_galois_key_method_I_stage_1(
        MultipartyGaloiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
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
            [&](Secretkey<Scheme::BFV>& sk_)
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
    HEMultiPartyManager<Scheme::BFV>::generate_galois_key_method_II_stage_1(
        MultipartyGaloiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
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

        DeviceVector<Data64> errors_a(2 * context_->Q_prime_size * context_->d *
                                          context_->n,
                                      options.stream_);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (context_->Q_prime_size * context_->d * context_->n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, options.stream_);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                a_poly, context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, context_->d, options.stream_);

        RNGSeed gen_seed1;
        RandomNumberGenerator::instance().set(gen_seed1.key_, gen_seed1.nonce_,
                                              gen_seed1.personalization_string_,
                                              options.stream_);

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
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
                                context_->Q_prime_size, context_->d,
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
                            context_->d * context_->Q_prime_size,
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
                            context_->Sk_pair_->data(), context_->n_power,
                            context_->Q_prime_size, context_->d,
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
                            context_->Q_prime_size, context_->d,
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
                        context_->d * context_->Q_prime_size,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((context_->n >> 8),
                                                   context_->Q_prime_size, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero, context_->Sk_pair_->data(),
                        context_->n_power, context_->Q_prime_size, context_->d,
                        context_->Q_size, context_->P_size);
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
                                context_->Q_prime_size, context_->d,
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
                            context_->d * context_->Q_prime_size,
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
                            context_->Sk_pair_->data(), context_->n_power,
                            context_->Q_prime_size, context_->d,
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
                            context_->Q_prime_size, context_->d,
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
                        context_->d * context_->Q_prime_size,
                        context_->Q_prime_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((context_->n >> 8),
                                                   context_->Q_prime_size, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        context_->modulus_->data(), context_->factor_->data(),
                        gk.galois_elt_zero, context_->Sk_pair_->data(),
                        context_->n_power, context_->Q_prime_size, context_->d,
                        context_->Q_size, context_->P_size);
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_galois_key_stage_2(
        std::vector<MultipartyGaloiskey<Scheme::BFV>>& all_gk,
        Galoiskey<Scheme::BFV>& gk, const ExecutionOptions& options)
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

    //

    __host__ void HEMultiPartyManager<Scheme::BFV>::partial_decrypt_stage_1(
        Ciphertext<Scheme::BFV>& ciphertext, Secretkey<Scheme::BFV>& sk,
        Ciphertext<Scheme::BFV>& partial_ciphertext, const cudaStream_t stream)
    {
        Data64* ct0 = ciphertext.data();
        Data64* ct1 =
            ciphertext.data() + (context_->Q_size << context_->n_power);

        DeviceVector<Data64> temp_memory(context_->n * context_->Q_size,
                                         stream);
        Data64* ct1_temp = temp_memory.data();

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!ciphertext.in_ntt_domain_)
        {
            gpuntt::GPU_NTT(ct1, ct1_temp, context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->Q_size, context_->Q_size);

            sk_multiplication<<<dim3((context_->n >> 8), context_->Q_size, 1),
                                256, 0, stream>>>(
                ct1_temp, sk.data(), ct1_temp, context_->modulus_->data(),
                context_->n_power, context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((context_->n >> 8), context_->Q_size, 1),
                                256, 0, stream>>>(
                ct1, sk.data(), ct1_temp, context_->modulus_->data(),
                context_->n_power, context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        gpuntt::GPU_INTT(
            ct1_temp, output_memory.data() + (context_->Q_size * context_->n),
            context_->intt_table_->data(), context_->modulus_->data(), cfg_intt,
            context_->Q_size, context_->Q_size);

        DeviceVector<Data64> error_poly(context_->Q_size * context_->n, stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly.data(), context_->modulus_->data(),
                context_->n_power, context_->Q_size, 1, stream);

        // TODO: Optimize it!
        addition<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                   stream>>>(
            output_memory.data() + (context_->Q_size * context_->n),
            error_poly.data(),
            output_memory.data() + (context_->Q_size * context_->n),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 1), 256, 0, stream>>>(
            ct0, output_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        partial_ciphertext.memory_set(std::move(output_memory));
    }

    __host__ void HEMultiPartyManager<Scheme::BFV>::partial_decrypt_stage_2(
        std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
        Plaintext<Scheme::BFV>& plaintext, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n, stream);

        int cipher_count = ciphertexts.size();

        DeviceVector<Data64> temp_sum(context_->Q_size << context_->n_power,
                                      stream);

        Data64* ct0 = ciphertexts[0].data();
        Data64* ct1 =
            ciphertexts[0].data() + (context_->Q_size << context_->n_power);
        addition<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                   stream>>>(ct0, ct1, temp_sum.data(),
                             context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            Data64* ct1_i =
                ciphertexts[i].data() + (context_->Q_size << context_->n_power);

            addition<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                       stream>>>(ct1_i, temp_sum.data(), temp_sum.data(),
                                 context_->modulus_->data(), context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        decryption_fusion_bfv_kernel<<<dim3((context_->n >> 8), 1, 1), 256, 0,
                                       stream>>>(
            temp_sum.data(), output_memory.data(), context_->modulus_->data(),
            context_->plain_modulus_, context_->gamma_, context_->Qi_t_->data(),
            context_->Qi_gamma_->data(), context_->Qi_inverse_->data(),
            context_->mulq_inv_t_, context_->mulq_inv_gamma_,
            context_->inv_gamma_, context_->n_power, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::distributed_bootstrapping_stage1(
        Ciphertext<Scheme::BFV>& common, Ciphertext<Scheme::BFV>& output,
        Secretkey<Scheme::BFV>& secret_key, const RNGSeed& seed,
        const cudaStream_t stream)
    {
        RNGSeed common_seed = seed;

        DeviceVector<Data64> temp_vector(4 * context_->Q_size * context_->n,
                                         stream);
        Data64* error_poly = temp_vector.data();
        Data64* a_poly = error_poly + (2 * context_->Q_size * context_->n);
        Data64* random_t_poly = a_poly + (context_->Q_size * context_->n);

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

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, context_->modulus_->data(),
                context_->n_power, context_->Q_size, 2, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                random_t_poly, context_->plain_modulus2_->data(),
                context_->n_power, 1, 1, stream);

        DeviceVector<Data64> output_memory((2 * context_->Q_size * context_->n),
                                           stream);

        Data64* ct0 = common.data();
        Data64* ct1 = common.data() + (context_->Q_size << context_->n_power);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!common.in_ntt_domain_)
        {
            DeviceVector<Data64> temp_memory(context_->n * context_->Q_size,
                                             stream);
            gpuntt::GPU_NTT(ct1, temp_memory.data(),
                            context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            context_->Q_size, context_->Q_size);

            col_boot_dec_mul_with_sk<<<dim3((context_->n >> 8),
                                            context_->Q_size, 2),
                                       256, 0, stream>>>(
                temp_memory.data(), a_poly, secret_key.data(),
                output_memory.data(), context_->modulus_->data(),
                context_->n_power, context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            col_boot_dec_mul_with_sk<<<dim3((context_->n >> 8),
                                            context_->Q_size, 2),
                                       256, 0, stream>>>(
                ct1, a_poly, secret_key.data(), output_memory.data(),
                context_->modulus_->data(), context_->n_power,
                context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                 context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 2 * context_->Q_size, context_->Q_size);

        col_boot_add_random_and_errors<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            output_memory.data(), error_poly, random_t_poly,
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power,
            context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::distributed_bootstrapping_stage2(
        std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
        Ciphertext<Scheme::BFV>& common, Ciphertext<Scheme::BFV>& output,
        const RNGSeed& seed, const cudaStream_t stream)
    {
        RNGSeed common_seed = seed;
        int cipher_count = ciphertexts.size();

        DeviceVector<Data64> h_memory((2 * context_->Q_size * context_->n),
                                      stream);

        global_memory_replace_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            ciphertexts[0].data(), h_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            addition<<<dim3((context_->n >> 8), context_->Q_size, 2), 256, 0,
                       stream>>>(ciphertexts[i].data(), h_memory.data(),
                                 h_memory.data(), context_->modulus_->data(),
                                 context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        DeviceVector<Data64> rand_message_memory(context_->n, stream);

        Data64* h0 = h_memory.data();
        Data64* h1 = h_memory.data() + (context_->Q_size << context_->n_power);

        decryption_kernel<<<dim3((context_->n >> 8), 1, 1), 256, 0, stream>>>(
            common.data(), h0, rand_message_memory.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->gamma_, context_->Qi_t_->data(),
            context_->Qi_gamma_->data(), context_->Qi_inverse_->data(),
            context_->mulq_inv_t_, context_->mulq_inv_gamma_,
            context_->inv_gamma_, context_->n_power, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //////////////////////////////

        DeviceVector<Data64> output_memory((2 * context_->Q_size * context_->n),
                                           stream);

        Data64* ct0 = output_memory.data();
        Data64* ct1 =
            output_memory.data() + (context_->Q_size << context_->n_power);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                ct1, context_->modulus_->data(), context_->n_power,
                context_->Q_size, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(ct1, context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 context_->Q_size, context_->Q_size);

        col_boot_enc<<<dim3((context_->n >> 8), context_->Q_size, 1), 256, 0,
                       stream>>>(
            ct0, h1, rand_message_memory.data(), context_->modulus_->data(),
            context_->plain_modulus_, context_->Q_mod_t_,
            context_->upper_threshold_, context_->coeeff_div_plainmod_->data(),
            context_->n_power, context_->Q_size);

        output.memory_set(std::move(output_memory));
    }

} // namespace heongpu
