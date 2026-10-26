// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <optional>
#include <heongpu/host/bfv/operator.cuh>

namespace heongpu
{
    __host__
    HEOperator<Scheme::BFV>::HEOperator(HEContext<Scheme::BFV> context,
                                        HEEncoder<Scheme::BFV>& encoder)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);
        encoding_location_ = encoder.encoding_location_;
    }

    __host__ void HEOperator<Scheme::BFV>::add(Ciphertext<Scheme::BFV>& input1,
                                               Ciphertext<Scheme::BFV>& input2,
                                               Ciphertext<Scheme::BFV>& output,
                                               const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        input2.validate_context(context_);

        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
                (cipher_size * context_->n * context_->Q_size) ||
            input2.memory_size() <
                (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::BFV>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::BFV>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::BFV>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * context_->n *
                                     context_->Q_size),
                                    options.stream_);

                                addition<<<dim3((context_->n >> 8),
                                                context_->Q_size, cipher_size),
                                           256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(),
                                    context_->modulus_->data(),
                                    context_->n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = context_->scheme_;
                                output_.context_id_ = context_->context_id_;
                                output_.ring_size_ = context_->n;
                                output_.coeff_modulus_count_ = context_->Q_size;
                                output_.cipher_size_ = cipher_size;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;
                                output_.ciphertext_generated_ = true;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator<Scheme::BFV>::sub(Ciphertext<Scheme::BFV>& input1,
                                               Ciphertext<Scheme::BFV>& input2,
                                               Ciphertext<Scheme::BFV>& output,
                                               const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        input2.validate_context(context_);

        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
                (cipher_size * context_->n * context_->Q_size) ||
            input2.memory_size() <
                (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::BFV>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::BFV>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::BFV>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * context_->n *
                                     context_->Q_size),
                                    options.stream_);

                                substraction<<<dim3((context_->n >> 8),
                                                    context_->Q_size,
                                                    cipher_size),
                                               256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(),
                                    context_->modulus_->data(),
                                    context_->n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = context_->scheme_;
                                output_.context_id_ = context_->context_id_;
                                output_.ring_size_ = context_->n;
                                output_.coeff_modulus_count_ = context_->Q_size;
                                output_.cipher_size_ = cipher_size;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;
                                output_.ciphertext_generated_ = true;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void
    HEOperator<Scheme::BFV>::negate(Ciphertext<Scheme::BFV>& input1,
                                    Ciphertext<Scheme::BFV>& output,
                                    const ExecutionOptions& options)
    {
        input1.validate_context(context_);

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients, got ", input1.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::BFV>& input1_)
            {
                output_storage_manager(
                    output,
                    [&](Ciphertext<Scheme::BFV>& output_)
                    {
                        DeviceVector<Data64> output_memory(
                            (cipher_size * context_->n * context_->Q_size),
                            options.stream_);

                        negation<<<dim3((context_->n >> 8), context_->Q_size,
                                        cipher_size),
                                   256, 0, options.stream_>>>(
                            input1_.data(), output_memory.data(),
                            context_->modulus_->data(), context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        output_.scheme_ = context_->scheme_;
                        output_.context_id_ = context_->context_id_;
                        output_.ring_size_ = context_->n;
                        output_.coeff_modulus_count_ = context_->Q_size;
                        output_.cipher_size_ = cipher_size;
                        output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                        output_.relinearization_required_ =
                            input1_.relinearization_required_;
                        output_.ciphertext_generated_ = true;

                        output_.memory_set(std::move(output_memory));
                    },
                    options);
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator<Scheme::BFV>::add_plain_bfv(
        Ciphertext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& input2,
        Ciphertext<Scheme::BFV>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < context_->n)
        {
            throw ParameterError(error_message(
                "Invalid Plaintext size! Expected at least ", context_->n,
                " coefficients, got ", input2.size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * context_->Q_size), stream);

        addition_plain_bfv_poly<<<dim3((context_->n >> 8), context_->Q_size,
                                       cipher_size),
                                  256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::add_plain_bfv_inplace(
        Ciphertext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& input2,
        const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < context_->n)
        {
            throw ParameterError(error_message(
                "Invalid Plaintext size! Expected at least ", context_->n,
                " coefficients, got ", input2.size(), "."));
        }

        addition_plain_bfv_poly_inplace<<<
            dim3((context_->n >> 8), context_->Q_size, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), input1.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::BFV>::sub_plain_bfv(
        Ciphertext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& input2,
        Ciphertext<Scheme::BFV>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < context_->n)
        {
            throw ParameterError(error_message(
                "Invalid Plaintext size! Expected at least ", context_->n,
                " coefficients, got ", input2.size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * context_->Q_size), stream);

        substraction_plain_bfv_poly<<<dim3((context_->n >> 8), context_->Q_size,
                                           cipher_size),
                                      256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::sub_plain_bfv_inplace(
        Ciphertext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& input2,
        const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * context_->Q_size),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < context_->n)
        {
            throw ParameterError(error_message(
                "Invalid Plaintext size! Expected at least ", context_->n,
                " coefficients, got ", input2.size(), "."));
        }

        substraction_plain_bfv_poly_inplace<<<
            dim3((context_->n >> 8), context_->Q_size, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), input1.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::BFV>::multiply_bfv(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& input2,
        Ciphertext<Scheme::BFV>& output, const cudaStream_t stream)
    {
        if ((input1.in_ntt_domain_ != false) ||
            (input2.in_ntt_domain_ != false))
        {
            throw std::invalid_argument("Ciphertexts should be in same domain");
        }

        if (input1.memory_size() < (2 * context_->n * context_->Q_size) ||
            input2.memory_size() < (2 * context_->n * context_->Q_size))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (2 * context_->n * context_->Q_size),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory((3 * context_->n * context_->Q_size),
                                           stream);

        DeviceVector<Data64> temp_mul(
            (4 * context_->n * (context_->bsk_modulus + context_->Q_size)) +
                (3 * context_->n * (context_->bsk_modulus + context_->Q_size)),
            stream);
        Data64* temp1_mul = temp_mul.data();
        Data64* temp2_mul =
            temp1_mul +
            (4 * context_->n * (context_->bsk_modulus + context_->Q_size));

        fast_convertion<<<dim3((context_->n >> 8), 4, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), temp1_mul, context_->modulus_->data(),
            context_->base_Bsk_->data(), context_->m_tilde_,
            context_->inv_prod_q_mod_m_tilde_,
            context_->inv_m_tilde_mod_Bsk_->data(),
            context_->prod_q_mod_Bsk_->data(),
            context_->base_change_matrix_Bsk_->data(),
            context_->base_change_matrix_m_tilde_->data(),
            context_->inv_punctured_prod_mod_base_array_->data(),
            context_->n_power, context_->Q_size, context_->bsk_modulus);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->q_Bsk_n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            temp1_mul, context_->q_Bsk_merge_ntt_tables_->data(),
            context_->q_Bsk_merge_modulus_->data(), cfg_ntt,
            ((context_->bsk_modulus + context_->Q_size) * 4),
            (context_->bsk_modulus + context_->Q_size));

        cross_multiplication<<<dim3((context_->n >> 8),
                                    (context_->bsk_modulus + context_->Q_size),
                                    1),
                               256, 0, stream>>>(
            temp1_mul,
            temp1_mul + (((context_->bsk_modulus + context_->Q_size) * 2) *
                         context_->n),
            temp2_mul, context_->q_Bsk_merge_modulus_->data(),
            context_->n_power, (context_->bsk_modulus + context_->Q_size));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_INTT_Inplace(
            temp2_mul, context_->q_Bsk_merge_intt_tables_->data(),
            context_->q_Bsk_merge_modulus_->data(), cfg_intt,
            (3 * (context_->bsk_modulus + context_->Q_size)),
            (context_->bsk_modulus + context_->Q_size));

        fast_floor<<<dim3((context_->n >> 8), 3, 1), 256, 0, stream>>>(
            temp2_mul, output_memory.data(), context_->modulus_->data(),
            context_->base_Bsk_->data(), context_->plain_modulus_,
            context_->inv_punctured_prod_mod_base_array_->data(),
            context_->base_change_matrix_Bsk_->data(),
            context_->inv_prod_q_mod_Bsk_->data(),
            context_->inv_punctured_prod_mod_B_array_->data(),
            context_->base_change_matrix_q_->data(),
            context_->base_change_matrix_msk_->data(),
            context_->inv_prod_B_mod_m_sk_, context_->prod_B_mod_q_->data(),
            context_->n_power, context_->Q_size, context_->bsk_modulus);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::multiply_plain_bfv(
        Ciphertext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& input2,
        Ciphertext<Scheme::BFV>& output, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        if (input1.in_ntt_domain_)
        {
            cipherplain_kernel<<<dim3((context_->n >> 8), context_->Q_size, 2),
                                 256, 0, stream>>>(
                input1.data(), input2.data(), output_memory.data(),
                context_->modulus_->data(), context_->n_power,
                context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_plain_mul(context_->n * context_->Q_size,
                                                stream);
            Data64* temp1_plain_mul = temp_plain_mul.data();

            threshold_kernel<<<dim3((context_->n >> 8), context_->Q_size, 1),
                               256, 0, stream>>>(
                input2.data(), temp1_plain_mul, context_->modulus_->data(),
                context_->upper_halfincrement_->data(),
                context_->upper_threshold_, context_->n_power,
                context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = context_->n_power,
                .ntt_type = gpuntt::FORWARD,
                .ntt_layout = gpuntt::PerPolynomial,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = context_->n_power,
                .ntt_type = gpuntt::INVERSE,
                .ntt_layout = gpuntt::PerPolynomial,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = context_->n_inverse_->data(),
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(temp1_plain_mul,
                                    context_->ntt_table_->data(),
                                    context_->modulus_->data(), cfg_ntt,
                                    context_->Q_size, context_->Q_size);

            gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                            context_->ntt_table_->data(),
                            context_->modulus_->data(), cfg_ntt,
                            2 * context_->Q_size, context_->Q_size);

            cipherplain_kernel<<<dim3((context_->n >> 8), context_->Q_size, 2),
                                 256, 0, stream>>>(
                output_memory.data(), temp1_plain_mul, output_memory.data(),
                context_->modulus_->data(), context_->n_power,
                context_->Q_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_INTT_Inplace(output_memory.data(),
                                     context_->intt_table_->data(),
                                     context_->modulus_->data(), cfg_intt,
                                     2 * context_->Q_size, context_->Q_size);
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator<Scheme::BFV>::relinearize_external_product_method2_inplace(
        Ciphertext<Scheme::BFV>& input1, Relinkey<Scheme::BFV>& relin_key,
        const cudaStream_t stream)
    {
        const int d = context_->d;

        DeviceVector<Data64> temp_relin(
            (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (context_->n * context_->Q_size *
                                             context_->Q_prime_size);

        decompose_to_Qtilde(input1.data() +
                                (context_->Q_size << (context_->n_power + 1)),
                            temp1_relin, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                d * context_->Q_prime_size,
                                context_->Q_prime_size);

        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp1_relin, relin_key.data(), temp2_relin,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp1_relin, key_location.data(), temp2_relin,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
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

        gpuntt::GPU_INTT_Inplace(temp2_relin, context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 2 * context_->Q_prime_size,
                                 context_->Q_prime_size);

        if (context_->P_size == 1)
        {
            divide_round_lastq_kernel<<<dim3((context_->n >> 8),
                                             context_->Q_size, 2),
                                        256, 0, stream>>>(
                temp2_relin, input1.data(), input1.data(),
                context_->modulus_->data(), context_->half_p_->data(),
                context_->half_mod_->data(), context_->last_q_modinv_->data(),
                context_->n_power, context_->Q_size);
        }
        else
        {
            divide_round_lastq_extended_kernel<<<dim3((context_->n >> 8),
                                                      context_->Q_size, 2),
                                                 256, 0, stream>>>(
                temp2_relin, input1.data(), input1.data(),
                context_->modulus_->data(), context_->half_p_->data(),
                context_->half_mod_->data(), context_->last_q_modinv_->data(),
                context_->n_power, context_->Q_prime_size, context_->Q_size,
                context_->P_size);
        }
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator<Scheme::BFV>::decompose_to_Qtilde(Data64* input, Data64* output,
                                                 cudaStream_t stream)
    {
        if (context_->P_size == 1)
        {
            // Every digit is one prime.
            single_prime_digits_to_Qtilde_kernel<<<
                dim3((context_->n >> 8), context_->d, 1), 256, 0, stream>>>(
                input, output, context_->modulus_->data(), context_->n_power,
                context_->Q_prime_size, context_->Q_size, 0);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            return;
        }

        base_conversion_DtoQtilde_relin_kernel<<<
            dim3((context_->n >> 8), context_->d, 1), 256, 0, stream>>>(
            input, output, context_->modulus_->data(),
            context_->base_change_matrix_D_to_Q_tilda_->data(),
            context_->Mi_inv_D_to_Q_tilda_->data(),
            context_->prod_D_to_Q_tilda_->data(), context_->I_j_->data(),
            context_->I_location_->data(), context_->n_power, context_->Q_size,
            context_->Q_prime_size, context_->d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::BFV>::rotate_method_II(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        Galoiskey<Scheme::BFV>& galois_key, int shift,
        const cudaStream_t stream)
    {
        int galoiselt =
            steps_to_galois_elt(shift, context_->n, galois_key.group_order_);
        bool key_exist = (galois_key.storage_type_ == storage_type::DEVICE)
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_method_II(input1, output, galois_key, galoiselt,
                                   stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw KeyNotFoundError(error_message(
                        "Galois key not present for rotation step ", index_in,
                        " (needed for shift ", shift,
                        ")! Generate the Galoiskey with this shift."));
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext<Scheme::BFV>* current_input = &input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_method_II(*current_input, output, galois_key,
                                       galois_elt, stream);
                current_input = &output;
            }
        }
    }

    __host__ std::vector<Ciphertext<Scheme::BFV>>
    HEOperator<Scheme::BFV>::rotate_rows_hoisted(
        Ciphertext<Scheme::BFV>& input1, Galoiskey<Scheme::BFV>& galois_key,
        const std::vector<int>& shifts, const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        galois_key.validate_context(context_);

        if (input1.relinearization_required_)
        {
            throw StateError("Ciphertext can not be rotated! "
                             "Relinearize it first.");
        }

        int current_decomp_count = context_->Q_size;

        if (input1.memory_size() < (2 * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (2 * context_->n * current_decomp_count), " coefficients, got ",
                input1.memory_size(), "."));
        }

        auto has_key = [&](int galois_elt)
        {
            return (galois_key.storage_type_ == storage_type::DEVICE)
                       ? (galois_key.device_location_.find(galois_elt) !=
                          galois_key.device_location_.end())
                       : (galois_key.host_location_.find(galois_elt) !=
                          galois_key.host_location_.end());
        };

        std::vector<Ciphertext<Scheme::BFV>> results;
        results.reserve(shifts.size());
        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::BFV>& input1_)
            {
                std::optional<GaloisDecomposition> decomposition;
                for (int shift : shifts)
                {
                    Ciphertext<Scheme::BFV> output(context_, options);
                    // Rotating by a multiple of the row size gives the input.
                    shift %= (context_->n >> 1);
                    int galois_elt =
                        (shift == 0)
                            ? 0
                            : steps_to_galois_elt(shift, context_->n,
                                                  galois_key.group_order_);
                    if (shift == 0)
                    {
                        output = input1;
                    }
                    else if (has_key(galois_elt))
                    {
                        if (!decomposition)
                        {
                            decomposition =
                                decompose_for_galois(input1_, options.stream_);
                        }
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::BFV>& output_)
                            {
                                apply_galois_decomposed(
                                    input1_, *decomposition, output_,
                                    galois_key, galois_elt, options.stream_);
                            },
                            options);
                    }
                    else
                    {
                        rotate_rows(input1_, output, galois_key, shift,
                                    options);
                    }
                    results.push_back(std::move(output));
                }
            },
            options, false);
        return results;
    }

    __host__ HEOperator<Scheme::BFV>::GaloisDecomposition
    HEOperator<Scheme::BFV>::decompose_for_galois(
        Ciphertext<Scheme::BFV>& input1, const cudaStream_t stream)
    {
        const int d = context_->d;

        GaloisDecomposition decomposition{DeviceVector<Data64>(
            context_->n * context_->Q_prime_size * d, stream)};

        decompose_to_Qtilde(input1.data() +
                                (context_->Q_size << context_->n_power),
                            decomposition.digits.data(), stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            decomposition.digits.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt, d * context_->Q_prime_size,
            context_->Q_prime_size);

        return decomposition;
    }

    __host__ void HEOperator<Scheme::BFV>::apply_galois_decomposed(
        Ciphertext<Scheme::BFV>& input1, GaloisDecomposition& decomposition,
        Ciphertext<Scheme::BFV>& output, Galoiskey<Scheme::BFV>& galois_key,
        int galois_elt, const cudaStream_t stream)
    {
        const int d = context_->d;

        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);
        DeviceVector<Data64> key_switched(
            2 * context_->n * context_->Q_prime_size, stream);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(decomposition.digits.data(),
                          galois_key.device_location_[galois_elt].data(),
                          key_switched.data(), context_->modulus_->data(),
                          context_->n_power, context_->Q_prime_size,
                          iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(decomposition.digits.data(), key_location.data(),
                          key_switched.data(), context_->modulus_->data(),
                          context_->n_power, context_->Q_prime_size,
                          iteration_count_1, iteration_count_2);
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

        gpuntt::GPU_INTT_Inplace(
            key_switched.data(), context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * context_->Q_prime_size,
            context_->Q_prime_size);

        // ModDown + Permute; c0 is read from input1, which output replaces
        // only afterwards.
        divide_round_lastq_permute_bfv_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            key_switched.data(), input1.data(), output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galois_elt, context_->n_power, context_->Q_prime_size,
            context_->Q_size, context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::apply_galois_method_II(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        Galoiskey<Scheme::BFV>& galois_key, int galois_elt,
        const cudaStream_t stream)
    {
        GaloisDecomposition decomposition =
            decompose_for_galois(input1, stream);
        apply_galois_decomposed(input1, decomposition, output, galois_key,
                                galois_elt, stream);
    }

    __host__ void HEOperator<Scheme::BFV>::rotate_columns_method_II(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        Galoiskey<Scheme::BFV>& galois_key, const cudaStream_t stream)
    {
        const int d = context_->d;

        int galoiselt = galois_key.galois_elt_zero;

        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_prime_size * d) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation + (2 * context_->n * context_->Q_prime_size * d);

        // c0 is read from input1, which is replaced only at the end.
        temp0_rotation = input1.data();

        decompose_to_Qtilde(input1.data() +
                                (context_->Q_size << context_->n_power),
                            temp2_rotation, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                d * context_->Q_prime_size,
                                context_->Q_prime_size);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp2_rotation, galois_key.c_data(), temp3_rotation,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(galois_key.zero_host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp2_rotation, key_location.data(), temp3_rotation,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
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

        gpuntt::GPU_INTT_Inplace(temp3_rotation, context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 2 * context_->Q_prime_size,
                                 context_->Q_prime_size);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galoiselt, context_->n_power, context_->Q_prime_size,
            context_->Q_size, context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::switchkey_method_II(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        Switchkey<Scheme::BFV>& switch_key, const cudaStream_t stream)
    {
        const int d = context_->d;

        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_prime_size * d) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation + (2 * context_->n * context_->Q_prime_size * d);

        // c0 and c1 are read from input1, which is replaced only at the end.
        temp0_rotation = input1.data();
        temp1_rotation =
            input1.data() + (context_->Q_size << context_->n_power);

        decompose_to_Qtilde(temp1_rotation, temp2_rotation, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                d * context_->Q_prime_size,
                                context_->Q_prime_size);

        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (switch_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp2_rotation, switch_key.data(), temp3_rotation,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((context_->n >> 8), context_->Q_prime_size, 1), 256, 0,
                stream>>>(temp2_rotation, key_location.data(), temp3_rotation,
                          context_->modulus_->data(), context_->n_power,
                          context_->Q_prime_size, iteration_count_1,
                          iteration_count_2);
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

        gpuntt::GPU_INTT_Inplace(temp3_rotation, context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt,
                                 2 * context_->Q_prime_size,
                                 context_->Q_prime_size);

        if (context_->P_size == 1)
        {
            divide_round_lastq_switchkey_kernel<<<dim3((context_->n >> 8),
                                                       context_->Q_size, 2),
                                                  256, 0, stream>>>(
                temp3_rotation, temp0_rotation, output_memory.data(),
                context_->modulus_->data(), context_->half_p_->data(),
                context_->half_mod_->data(), context_->last_q_modinv_->data(),
                context_->n_power, context_->Q_size);
        }
        else
        {
            divide_round_lastq_extended_switchkey_kernel<<<
                dim3((context_->n >> 8), context_->Q_size, 2), 256, 0,
                stream>>>(temp3_rotation, temp0_rotation, output_memory.data(),
                          context_->modulus_->data(), context_->half_p_->data(),
                          context_->half_mod_->data(),
                          context_->last_q_modinv_->data(), context_->n_power,
                          context_->Q_prime_size, context_->Q_size,
                          context_->P_size);
        }
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::negacyclic_shift_poly_coeffmod(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        int index, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        DeviceVector<Data64> temp(2 * context_->n * context_->Q_size, stream);

        negacyclic_shift_poly_coeffmod_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            input1.data(), temp.data(), context_->modulus_->data(), index,
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: do with efficient way!
        global_memory_replace_kernel<<<
            dim3((context_->n >> 8), context_->Q_size, 2), 256, 0, stream>>>(
            temp.data(), output_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::transform_to_ntt_bfv_plain(
        Plaintext<Scheme::BFV>& input1, Plaintext<Scheme::BFV>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<Data64> temp_plain_mul(context_->n * context_->Q_size,
                                            stream);
        Data64* temp1_plain_mul = temp_plain_mul.data();

        threshold_kernel<<<dim3((context_->n >> 8), context_->Q_size, 1), 256,
                           0, stream>>>(
            input1.data(), temp1_plain_mul, context_->modulus_->data(),
            context_->upper_halfincrement_->data(), context_->upper_threshold_,
            context_->n_power, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(temp1_plain_mul, output_memory.data(),
                        context_->ntt_table_->data(),
                        context_->modulus_->data(), cfg_ntt, context_->Q_size,
                        context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::transform_to_ntt_bfv_cipher(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                        context_->ntt_table_->data(),
                        context_->modulus_->data(), cfg_ntt,
                        2 * context_->Q_size, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::BFV>::transform_from_ntt_bfv_cipher(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * context_->n * context_->Q_size),
                                           stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), output_memory.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * context_->Q_size, context_->Q_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    //                       BOOTSRAPPING                         //
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    __host__ Ciphertext<Scheme::BFV>
    HEOperator<Scheme::BFV>::operator_from_ciphertext(
        Ciphertext<Scheme::BFV>& input, cudaStream_t stream)
    {
        Ciphertext<Scheme::BFV> cipher;

        cipher.coeff_modulus_count_ = input.coeff_modulus_count_;
        cipher.cipher_size_ = input.cipher_size_;
        cipher.ring_size_ = input.ring_size_;

        cipher.scheme_ = input.scheme_;
        cipher.context_id_ = input.context_id_;
        cipher.in_ntt_domain_ = input.in_ntt_domain_;

        cipher.storage_type_ = storage_type::DEVICE;

        cipher.relinearization_required_ = input.relinearization_required_;
        cipher.ciphertext_generated_ = true;

        int cipher_memory_size = 2 * context_->Q_size * context_->n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    HEArithmeticOperator<Scheme::BFV>::HEArithmeticOperator(
        HEContext<Scheme::BFV> context, HEEncoder<Scheme::BFV>& encoder)
        : HEOperator<Scheme::BFV>(context, encoder)
    {
    }

    HELogicOperator<Scheme::BFV>::HELogicOperator(
        HEContext<Scheme::BFV> context, HEEncoder<Scheme::BFV>& encoder)
        : HEOperator<Scheme::BFV>(context, encoder)
    {
        // TODO: make it efficinet
        Data64 constant_1 = 1ULL;
        encoded_constant_one_ = DeviceVector<Data64>(context_->n);
        fill_device_vector<<<dim3((context_->n >> 8), 1, 1), 256>>>(
            encoded_constant_one_.data(), constant_1, context_->n);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_plain_inverse_->data(),
            .stream = 0};

        gpuntt::GPU_INTT_Inplace(
            encoded_constant_one_.data(), context_->plain_intt_tables_->data(),
            context_->plain_modulus2_->data(), cfg_intt, 1, 1);
    }

    __host__ void HELogicOperator<Scheme::BFV>::one_minus_cipher(
        Ciphertext<Scheme::BFV>& input1, Ciphertext<Scheme::BFV>& output,
        const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        addition_plain_bfv_poly<<<dim3((context_->n >> 8), context_->Q_size, 2),
                                  256, 0, options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), output.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HELogicOperator<Scheme::BFV>::one_minus_cipher_inplace(
        Ciphertext<Scheme::BFV>& input1, const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        addition_plain_bfv_poly_inplace<<<dim3((context_->n >> 8),
                                               context_->Q_size, 1),
                                          256, 0, options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), input1.data(),
            context_->modulus_->data(), context_->plain_modulus_,
            context_->Q_mod_t_, context_->upper_threshold_,
            context_->coeeff_div_plainmod_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
