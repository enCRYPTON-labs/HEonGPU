// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <NTL/ZZ.h>
#include <NTL/RR.h>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/cosine_approx.cuh>

namespace heongpu
{
    __host__
    HEOperator<Scheme::CKKS>::HEOperator(HEContext<Scheme::CKKS> context,
                                         HEEncoder<Scheme::CKKS>& encoder)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);

        std::vector<int> prime_loc;
        std::vector<int> input_loc;

        int counter = context_->Q_size;
        for (int i = 0; i < context_->Q_size; i++)
        {
            for (int j = 0; j < counter; j++)
            {
                prime_loc.push_back(j);
            }
            counter--;
            for (int j = 0; j < context_->P_size; j++)
            {
                prime_loc.push_back(context_->Q_size + j);
            }
        }

        counter = context_->Q_prime_size;
        for (int i = 0; i < context_->Q_prime_size - 1; i++)
        {
            int sum = counter - 1;
            for (int j = 0; j < 2; j++)
            {
                input_loc.push_back(sum);
                sum += counter;
            }
            counter--;
        }

        new_prime_locations_ = DeviceVector<int>(prime_loc);
        new_input_locations_ = DeviceVector<int>(input_loc);
        new_prime_locations = new_prime_locations_.data();
        new_input_locations = new_input_locations_.data();

        // Encode params
        slot_count_ = encoder.slot_count_;
        log_slot_count_ = encoder.log_slot_count_;
        two_pow_64_ = encoder.two_pow_64;
        reverse_order_ = encoder.reverse_order;
        special_ifft_roots_table_ = encoder.special_ifft_roots_table_;
    }

    __host__ void HEOperator<Scheme::CKKS>::add(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        input2.validate_context(context_);

        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should have same encoding type");
        }

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

        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
                (cipher_size * context_->n * current_decomp_count) ||
            input2.memory_size() <
                (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::CKKS>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::CKKS>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::CKKS>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * context_->n *
                                     current_decomp_count),
                                    options.stream_);

                                addition<<<dim3((context_->n >> 8),
                                                current_decomp_count,
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
                                output_.depth_ = input1_.depth_;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.encoding_ = input1_.encoding_;
                                output_.scale_ = input1_.scale_;
                                output_.rescale_required_ =
                                    (input1_.rescale_required_ ||
                                     input2_.rescale_required_);
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

    __host__ void HEOperator<Scheme::CKKS>::sub(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        input2.validate_context(context_);

        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should have same encoding type");
        }

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

        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
                (cipher_size * context_->n * current_decomp_count) ||
            input2.memory_size() <
                (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::CKKS>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::CKKS>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::CKKS>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * context_->n *
                                     current_decomp_count),
                                    options.stream_);

                                substraction<<<dim3((context_->n >> 8),
                                                    current_decomp_count,
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
                                output_.depth_ = input1_.depth_;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.encoding_ = input1_.encoding_;
                                output_.scale_ = input1_.scale_;
                                output_.rescale_required_ =
                                    (input1_.rescale_required_ ||
                                     input2_.rescale_required_);
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
    HEOperator<Scheme::CKKS>::negate(Ciphertext<Scheme::CKKS>& input1,
                                     Ciphertext<Scheme::CKKS>& output,
                                     const ExecutionOptions& options)
    {
        input1.validate_context(context_);

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::CKKS>& input1_)
            {
                output_storage_manager(
                    output,
                    [&](Ciphertext<Scheme::CKKS>& output_)
                    {
                        DeviceVector<Data64> output_memory(
                            (cipher_size * context_->n * current_decomp_count),
                            options.stream_);

                        negation<<<dim3((context_->n >> 8),
                                        current_decomp_count, cipher_size),
                                   256, 0, options.stream_>>>(
                            input1_.data(), output_memory.data(),
                            context_->modulus_->data(), context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        output_.scheme_ = context_->scheme_;
                        output_.context_id_ = context_->context_id_;
                        output_.ring_size_ = context_->n;
                        output_.coeff_modulus_count_ = context_->Q_size;
                        output_.cipher_size_ = cipher_size;
                        output_.depth_ = input1_.depth_;
                        output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.encoding_ = input1_.encoding_;
                        output_.scale_ = input1_.scale_;
                        output_.rescale_required_ = input1_.rescale_required_;
                        output_.relinearization_required_ =
                            input1_.relinearization_required_;
                        output_.ciphertext_generated_ = true;

                        output_.memory_set(std::move(output_memory));
                    },
                    options);
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator<Scheme::CKKS>::add_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Ciphertext and Plaintext should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < (context_->n * current_decomp_count))
        {
            throw ParameterError(
                error_message("Invalid Plaintext size! Expected at least ",
                              (context_->n * current_decomp_count),
                              " coefficients, got ", input2.size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        addition_plain_ckks_poly<<<dim3((context_->n >> 8),
                                        current_decomp_count, cipher_size),
                                   256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::add_plain_ckks_inplace(
        Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
        const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Ciphertext and Plaintext should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < (context_->n * current_decomp_count))
        {
            throw ParameterError(
                error_message("Invalid Plaintext size! Expected at least ",
                              (context_->n * current_decomp_count),
                              " coefficients, got ", input2.size(), "."));
        }

        addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256, 0,
                   stream>>>(input1.data(), input2.data(), input1.data(),
                             context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::add_constant_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, double input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        addition_constant_plain_ckks_poly<<<
            dim3((context_->n >> 8), current_decomp_count, cipher_size), 256, 0,
            stream>>>(input1.data(), input2, output_memory.data(),
                      context_->modulus_->data(), two_pow_64_,
                      context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::add_constant_plain_ckks_inplace(
        Ciphertext<Scheme::CKKS>& input1, double input2,
        const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        addition_constant_plain_ckks_poly<<<dim3((context_->n >> 8),
                                                 current_decomp_count, 1),
                                            256, 0, stream>>>(
            input1.data(), input2, input1.data(), context_->modulus_->data(),
            two_pow_64_, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::sub_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Ciphertext and Plaintext should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < (context_->n * current_decomp_count))
        {
            throw ParameterError(
                error_message("Invalid Plaintext size! Expected at least ",
                              (context_->n * current_decomp_count),
                              " coefficients, got ", input2.size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        substraction_plain_ckks_poly<<<dim3((context_->n >> 8),
                                            current_decomp_count, cipher_size),
                                       256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::sub_plain_ckks_inplace(
        Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
        const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Ciphertext and Plaintext should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        if (input2.size() < (context_->n * current_decomp_count))
        {
            throw ParameterError(
                error_message("Invalid Plaintext size! Expected at least ",
                              (context_->n * current_decomp_count),
                              " coefficients, got ", input2.size(), "."));
        }

        substraction<<<dim3((context_->n >> 8), current_decomp_count, 1), 256,
                       0, stream>>>(input1.data(), input2.data(), input1.data(),
                                    context_->modulus_->data(),
                                    context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::sub_constant_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, double input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        substraction_constant_plain_ckks_poly<<<
            dim3((context_->n >> 8), current_decomp_count, cipher_size), 256, 0,
            stream>>>(input1.data(), input2, output_memory.data(),
                      context_->modulus_->data(), two_pow_64_,
                      context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::sub_constant_plain_ckks_inplace(
        Ciphertext<Scheme::CKKS>& input1, double input2,
        const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        substraction_constant_plain_ckks_poly<<<dim3((context_->n >> 8),
                                                     current_decomp_count, 1),
                                                256, 0, stream>>>(
            input1.data(), input2, input1.data(), context_->modulus_->data(),
            two_pow_64_, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::add_constant_plain_complex_ckks(
        Ciphertext<Scheme::CKKS>& input1, Complex64 c,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;
        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        double scaled_real = c.real() * input1.scale_;
        double scaled_imag = c.imag() * input1.scale_;

        // Use NTL big integer to handle overflow
        NTL::ZZ real_zz;
        NTL::conv(real_zz, std::round(scaled_real));
        NTL::ZZ imag_zz;
        NTL::conv(imag_zz, std::round(scaled_imag));

        // Compute RNS representation
        std::vector<Data64> real_rns_host(current_decomp_count);
        std::vector<Data64> imag_rns_host(current_decomp_count);

        for (int i = 0; i < current_decomp_count; i++)
        {
            Data64 qi = context_->prime_vector_[i].value;
            NTL::ZZ qi_zz;
            NTL::conv(qi_zz, static_cast<long>(qi));

            // Compute (real_zz % qi)
            NTL::ZZ real_mod = real_zz % qi_zz;
            if (real_mod < 0)
            {
                real_mod += qi_zz;
            }
            real_rns_host[i] = NTL::to_long(real_mod);

            // Compute (imag_zz % qi)
            NTL::ZZ imag_mod = imag_zz % qi_zz;
            if (imag_mod < 0)
            {
                imag_mod += qi_zz;
            }
            imag_rns_host[i] = NTL::to_long(imag_mod);
        }

        DeviceVector<Data64> real_rns = DeviceVector<Data64>(real_rns_host);
        DeviceVector<Data64> imag_rns = DeviceVector<Data64>(imag_rns_host);

        cipher_add_by_gaussian_integer_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, cipher_size), 256, 0,
            stream>>>(input1.data(), real_rns.data(), imag_rns.data(),
                      output_memory.data(), context_->ntt_table_->data(),
                      context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::multiply_const_plain_complex_ckks(
        Ciphertext<Scheme::CKKS>& input1, Complex64 c,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;
        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        // Determine scaling factor based on whether the constant has a
        // fractional part
        double scale_factor = 1.0;
        double c_real = c.real();
        double c_imag = c.imag();

        // Check if we need to scale by modulus to preserve fractional parts
        if (c_real != 0)
        {
            int64_t value_int = static_cast<int64_t>(c_real);
            double value_float = c_real - static_cast<double>(value_int);
            if (value_float != 0)
            {
                scale_factor = static_cast<double>(
                    context_->prime_vector_[input1.level()].value);
            }
        }
        if (c_imag != 0)
        {
            int64_t value_int = static_cast<int64_t>(c_imag);
            double value_float = c_imag - static_cast<double>(value_int);
            if (value_float != 0)
            {
                scale_factor = static_cast<double>(
                    context_->prime_vector_[input1.level()].value);
            }
        }

        double scaled_real = c_real * scale_factor;
        double scaled_imag = c_imag * scale_factor;

        // Use NTL big integer to handle overflow
        NTL::ZZ real_zz;
        NTL::conv(real_zz, std::round(scaled_real));
        NTL::ZZ imag_zz;
        NTL::conv(imag_zz, std::round(scaled_imag));

        // Compute RNS representation
        std::vector<Data64> real_rns_host(current_decomp_count);
        std::vector<Data64> imag_rns_host(current_decomp_count);

        for (int i = 0; i < current_decomp_count; i++)
        {
            Data64 qi = context_->prime_vector_[i].value;
            NTL::ZZ qi_zz;
            NTL::conv(qi_zz, static_cast<long>(qi));

            // Compute (real_zz % qi)
            NTL::ZZ real_mod = real_zz % qi_zz;
            if (real_mod < 0)
            {
                real_mod += qi_zz;
            }
            real_rns_host[i] = NTL::to_long(real_mod);

            // Compute (imag_zz % qi)
            NTL::ZZ imag_mod = imag_zz % qi_zz;
            if (imag_mod < 0)
            {
                imag_mod += qi_zz;
            }
            imag_rns_host[i] = NTL::to_long(imag_mod);
        }

        DeviceVector<Data64> real_rns = DeviceVector<Data64>(real_rns_host);
        DeviceVector<Data64> imag_rns = DeviceVector<Data64>(imag_rns_host);

        cipher_mult_by_gaussian_integer_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, cipher_size), 256, 0,
            stream>>>(input1.data(), real_rns.data(), imag_rns.data(),
                      output_memory.data(), context_->ntt_table_->data(),
                      context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scale_ = input1.scale_ * scale_factor;
        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::scale_up_ckks(
        Ciphertext<Scheme::CKKS>& input, double scale,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        uint64_t scale_uint = static_cast<uint64_t>(scale);
        double scale_double = static_cast<double>(scale_uint);

        // Multiply by the scale constant (as a real number, no imaginary part)
        Complex64 scale_complex(scale_double, 0.0);
        multiply_const_plain_complex_ckks(input, scale_complex, output, stream);

        // Set scale to input.scale * scale (using original scale, not
        // scale_uint)
        output.scale_ = input.scale_ * scale;
    }

    __host__ void
    HEOperator<Scheme::CKKS>::mult_i_ckks(Ciphertext<Scheme::CKKS>& input1,
                                          Ciphertext<Scheme::CKKS>& output,
                                          const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;
        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        cipher_mult_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                       cipher_size),
                                  256, 0, stream>>>(
            input1.data(), output_memory.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator<Scheme::CKKS>::div_i_ckks(Ciphertext<Scheme::CKKS>& input1,
                                         Ciphertext<Scheme::CKKS>& output,
                                         const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;
        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() <
            (cipher_size * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (cipher_size * context_->n * current_decomp_count),
                " coefficients, got ", input1.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * context_->n * current_decomp_count), stream);

        cipher_div_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                      cipher_size),
                                 256, 0, stream>>>(
            input1.data(), output_memory.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::multiply_ckks(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        if (input1.memory_size() < (2 * context_->n * current_decomp_count) ||
            input2.memory_size() < (2 * context_->n * current_decomp_count))
        {
            throw ParameterError(error_message(
                "Invalid Ciphertexts size! Expected at least ",
                (2 * context_->n * current_decomp_count),
                " coefficients per ciphertext, got ", input1.memory_size(),
                " and ", input2.memory_size(), "."));
        }

        DeviceVector<Data64> output_memory(
            (3 * context_->n * current_decomp_count), stream);

        cross_multiplication<<<dim3((context_->n >> 8), (current_decomp_count),
                                    1),
                               256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));

        if (context_->scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::multiply_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
        Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled are not equal (depth ", input1.depth_,
                " and ", input2.depth_,
                ")! Use mod_drop to bring both operands to the same depth."));
        }

        if (input1.encoding_ != input2.encoding_)
        {
            throw std::invalid_argument(
                "Ciphertext and Plaintext should have same encoding type");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;
        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        cipherplain_multiplication_kernel<<<dim3((context_->n >> 8),
                                                 current_decomp_count, 2),
                                            256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (context_->scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::multiply_const_plain_ckks(
        Ciphertext<Scheme::CKKS>& input1, double input2,
        Ciphertext<Scheme::CKKS>& output, double scale,
        const cudaStream_t stream)
    {
        int current_decomp_count = context_->Q_size - input1.depth_;
        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        double value = input2 * scale;

        cipher_constant_plain_multiplication_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(input1.data(), value, output_memory.data(),
                      context_->modulus_->data(), two_pow_64_,
                      context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (context_->scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * scale;
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator<Scheme::CKKS>::relinearize_seal_method_inplace_ckks(
        Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
        const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(
            input1.data() + (current_decomp_count << (context_->n_power + 1)),
            context_->intt_table_->data(), context_->modulus_->data(), cfg_intt,
            current_decomp_count, current_decomp_count);

        DeviceVector<Data64> temp_relin(
            (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (context_->n * context_->Q_size *
                                             context_->Q_prime_size);

        cipher_broadcast_leveled_kernel<<<dim3((context_->n >> 8),
                                               current_decomp_count, 1),
                                          256, 0, stream>>>(
            input1.data() + (current_decomp_count << (context_->n_power + 1)),
            temp1_relin, context_->modulus_->data(), first_rns_mod_count,
            current_rns_mod_count, context_->n_power);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        int iteration_count_1 = current_decomp_count / 4;
        int iteration_count_2 = current_decomp_count % 4;
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp1_relin, relin_key.data(), temp2_relin,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp1_relin, key_location.data(), temp2_relin,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt2 = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data() + first_decomp_count,
            .stream = stream};

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            temp2_relin,
            context_->intt_table_->data() +
                (first_decomp_count << context_->n_power),
            context_->modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<
            dim3((context_->n >> 8), 2, 1), 256, 0, stream>>>(
            temp2_relin, temp1_relin, context_->modulus_->data(),
            context_->half_p_->data(), context_->half_mod_->data(),
            context_->n_power, first_decomp_count, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp1_relin, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        divide_round_lastq_leveled_stage_two_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(temp1_relin, temp2_relin, input1.data(), input1.data(),
                      context_->modulus_->data(),
                      context_->last_q_modinv_->data(), context_->n_power,
                      current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator<Scheme::CKKS>::relinearize_external_product_method2_inplace_ckks(
        Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
        const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        gpuntt::GPU_INTT_Inplace(
            input1.data() + (current_decomp_count << (context_->n_power + 1)),
            context_->intt_table_->data(), context_->modulus_->data(), cfg_intt,
            current_decomp_count, current_decomp_count);

        DeviceVector<Data64> temp_relin(
            (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (context_->n * context_->Q_size *
                                             context_->Q_prime_size);

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((context_->n >> 8),
                 context_->d_leveled->operator[](input1.depth_), 1),
            256, 0, stream>>>(
            input1.data() + (current_decomp_count << (context_->n_power + 1)),
            temp1_relin, context_->modulus_->data(),
            context_->base_change_matrix_D_to_Qtilda_leveled
                ->
                operator[](input1.depth_)
                .data(),
            context_->Mi_inv_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->prod_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->I_j_leveled->operator[](input1.depth_).data(),
            context_->I_location_leveled->operator[](input1.depth_).data(),
            context_->n_power, context_->d_leveled->operator[](input1.depth_),
            current_rns_mod_count, current_decomp_count, input1.depth_,
            context_->prime_location_leveled->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            context_->d_leveled->operator[](input1.depth_) *
                current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        int iteration_count_1 =
            context_->d_leveled->operator[](input1.depth_) / 4;
        int iteration_count_2 =
            context_->d_leveled->operator[](input1.depth_) % 4;
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp1_relin, relin_key.data(), temp2_relin,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp1_relin, key_location.data(), temp2_relin,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_relin, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        divide_round_lastq_extended_leveled_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(temp2_relin, temp1_relin, context_->modulus_->data(),
                      context_->half_p_->data(), context_->half_mod_->data(),
                      context_->last_q_modinv_->data(), context_->n_power,
                      current_rns_mod_count, current_decomp_count,
                      first_rns_mod_count, first_decomp_count,
                      context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp1_relin, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        addition<<<dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
                   stream>>>(temp1_relin, input1.data(), input1.data(),
                             context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::rescale_inplace_ckks_leveled(
        Ciphertext<Scheme::CKKS>& input1, const cudaStream_t stream)
    {
        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse =
                context_->n_inverse_->data() + (current_decomp_count - 1),
            .stream = stream};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        // int counter = first_rns_mod_count - 2;
        int counter = first_decomp_count - 1;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        DeviceVector<Data64> temp_rescale(
            (2 * context_->n * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);
        Data64* temp1_rescale = temp_rescale.data();
        Data64* temp2_rescale =
            temp1_rescale + (2 * context_->n * context_->Q_prime_size);

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            input1.data(),
            context_->intt_table_->data() +
                ((current_decomp_count - 1) << context_->n_power),
            context_->modulus_->data() + (current_decomp_count - 1), cfg_intt,
            2, 1,
            new_input_locations + ((input1.depth_ + context_->P_size) * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<
            dim3((context_->n >> 8), 2, 1), 256, 0, stream>>>(
            input1.data(), temp1_rescale, context_->modulus_->data(),
            context_->rescaled_half_->data() + input1.depth_,
            context_->rescaled_half_mod_->data() + location, context_->n_power,
            current_decomp_count - 1, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp1_rescale, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * (current_decomp_count - 1),
                                (current_decomp_count - 1));

        move_cipher_leveled_kernel<<<dim3((context_->n >> 8),
                                          current_decomp_count - 1, 2),
                                     256, 0, stream>>>(
            input1.data(), temp2_rescale, context_->n_power,
            current_decomp_count - 1);

        divide_round_lastq_rescale_kernel<<<dim3((context_->n >> 8),
                                                 current_decomp_count - 1, 2),
                                            256, 0, stream>>>(
            temp1_rescale, temp2_rescale, input1.data(),
            context_->modulus_->data(),
            context_->rescaled_last_q_modinv_->data() + location,
            context_->n_power, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (context_->scheme_ == scheme_type::ckks)
        {
            input1.scale_ =
                input1.scale_ /
                static_cast<double>(
                    context_->prime_vector_[current_decomp_count - 1].value);
        }

        input1.depth_++;
    }

    __host__ void HEOperator<Scheme::CKKS>::mod_drop_ckks_leveled_inplace(
        Ciphertext<Scheme::CKKS>& input1, const cudaStream_t stream)
    {
        if (input1.depth_ >= (context_->Q_size - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;

        int offset1 = current_decomp_count << context_->n_power;
        int offset2 = (current_decomp_count - 1) << context_->n_power;

        DeviceVector<Data64> temp_mod_drop_(context_->n * context_->Q_size,
                                            stream);
        Data64* temp_mod_drop = temp_mod_drop_.data();

        // TODO: do with efficient way!
        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            current_decomp_count - 1, 1),
                                       256, 0, stream>>>(
            input1.data() + offset1, temp_mod_drop, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            current_decomp_count - 1, 1),
                                       256, 0, stream>>>(
            temp_mod_drop, input1.data() + offset2, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input1.depth_++;
    }

    __host__ void HEOperator<Scheme::CKKS>::mod_drop_ckks_leveled(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        const cudaStream_t stream)
    {
        if (input1.depth_ >= (context_->Q_size - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;
        DeviceVector<Data64> output_memory(
            (current_decomp_count * context_->n * current_decomp_count),
            stream);

        global_memory_replace_offset_kernel<<<dim3((context_->n >> 8),
                                                   current_decomp_count - 1, 2),
                                              256, 0, stream>>>(
            input1.data(), output_memory.data(), current_decomp_count,
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::mod_drop_ckks_plaintext(
        Plaintext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& output,
        const cudaStream_t stream)
    {
        if (input1.depth_ >= (context_->Q_size - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        int current_decomp_count = context_->Q_size - input1.depth_;
        DeviceVector<Data64> output_memory(
            context_->n * (current_decomp_count - 1), stream);

        global_memory_replace_kernel<<<dim3((context_->n >> 8),
                                            current_decomp_count - 1, 1),
                                       256, 0, stream>>>(
            input1.data(), output_memory.data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.depth_ = input1.depth_ + 1;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::mod_drop_ckks_plaintext_inplace(
        Plaintext<Scheme::CKKS>& input1, const cudaStream_t stream)
    {
        if (input1.depth_ >= (context_->Q_size - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        input1.depth_++;
    }

    __host__ void HEOperator<Scheme::CKKS>::rotate_ckks_method_I(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& galois_key, int shift,
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
            apply_galois_ckks_method_I(input1, output, galois_key, galoiselt,
                                       stream);
        }
        else
        {
            std::vector<int> indexs = rotation_index_generator(
                shift, galois_key.max_log_slot_, galois_key.max_shift_);
            if (indexs.empty())
            {
                if (shift % (context_->n >> 1) == 0)
                {
                    // A multiple of the slot count does not move the slots.
                    output = input1;
                    return;
                }
                throw KeyNotFoundError(error_message(
                    "Galois key not present for shift ", shift,
                    "! The Galoiskey has neither a key for this shift nor the "
                    "power of two keys to build it from."));
            }
            std::vector<int> required_galoiselt;
            for (int index : indexs)
            {
                if (!(galois_key.galois_elt.find(index) !=
                      galois_key.galois_elt.end()))
                {
                    throw KeyNotFoundError(error_message(
                        "Galois key not present for rotation step ", index,
                        " (needed for shift ", shift,
                        ")! Generate the Galoiskey with this shift."));
                }
                required_galoiselt.push_back(galois_key.galois_elt[index]);
            }

            Ciphertext<Scheme::CKKS>* current_input = &input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_ckks_method_I(*current_input, output, galois_key,
                                           galois_elt, stream);
                current_input = &output;
            }
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::rotate_ckks_method_II(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& galois_key, int shift,
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
            apply_galois_ckks_method_II(input1, output, galois_key, galoiselt,
                                        stream);
        }
        else
        {
            std::vector<int> indexs = rotation_index_generator(
                shift, galois_key.max_log_slot_, galois_key.max_shift_);
            if (indexs.empty())
            {
                if (shift % (context_->n >> 1) == 0)
                {
                    // A multiple of the slot count does not move the slots.
                    output = input1;
                    return;
                }
                throw KeyNotFoundError(error_message(
                    "Galois key not present for shift ", shift,
                    "! The Galoiskey has neither a key for this shift nor the "
                    "power of two keys to build it from."));
            }
            std::vector<int> required_galoiselt;
            for (int index : indexs)
            {
                if (!(galois_key.galois_elt.find(index) !=
                      galois_key.galois_elt.end()))
                {
                    throw KeyNotFoundError(error_message(
                        "Galois key not present for rotation step ", index,
                        " (needed for shift ", shift,
                        ")! Generate the Galoiskey with this shift."));
                }
                required_galoiselt.push_back(galois_key.galois_elt[index]);
            }

            Ciphertext<Scheme::CKKS>* current_input = &input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_ckks_method_II(*current_input, output, galois_key,
                                            galois_elt, stream);
                current_input = &output;
            }
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::apply_galois_ckks_method_I(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
        const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation +
            (context_->n * context_->Q_size * context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        // TODO: make it efficient
        ckks_duplicate_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                     1),
                                256, 0, stream>>>(
            temp0_rotation, temp2_rotation, context_->modulus_->data(),
            context_->n_power, first_rns_mod_count, current_rns_mod_count,
            current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = current_decomp_count / 4;
        int iteration_count_2 = current_decomp_count % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, context_->modulus_->data(), first_rns_mod_count,
                current_decomp_count, iteration_count_1, iteration_count_2,
                context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp2_rotation, key_location.data(), temp3_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<dim3((context_->n >> 8),
                                                      current_decomp_count, 2),
                                                 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galois_elt, context_->n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));

        output.scheme_ = context_->scheme_;
        output.context_id_ = context_->context_id_;
        output.ring_size_ = context_->n;
        output.coeff_modulus_count_ = context_->Q_size;
        output.cipher_size_ = 2;
        output.depth_ = input1.depth_;
        output.scale_ = input1.scale_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.encoding_ = input1.encoding_;
        output.rescale_required_ = input1.rescale_required_;
        output.relinearization_required_ = input1.relinearization_required_;
        output.ciphertext_generated_ = true;
    }

    __host__ void HEOperator<Scheme::CKKS>::apply_galois_ckks_method_II(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
        const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size) +
                (2 * context_->n * context_->d_leveled->operator[](0) *
                 context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation + (context_->n * context_->Q_size);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * context_->n * context_->d_leveled->operator[](0) *
             context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((context_->n >> 8),
                 context_->d_leveled->operator[](input1.depth_), 1),
            256, 0, stream>>>(
            temp0_rotation + (current_decomp_count << context_->n_power),
            temp3_rotation, context_->modulus_->data(),
            context_->base_change_matrix_D_to_Qtilda_leveled
                ->
                operator[](input1.depth_)
                .data(),
            context_->Mi_inv_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->prod_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->I_j_leveled->operator[](input1.depth_).data(),
            context_->I_location_leveled->operator[](input1.depth_).data(),
            context_->n_power, context_->d_leveled->operator[](input1.depth_),
            current_rns_mod_count, current_decomp_count, input1.depth_,
            context_->prime_location_leveled->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            context_->d_leveled->operator[](input1.depth_) *
                current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 =
            context_->d_leveled->operator[](input1.depth_) / 4;
        int iteration_count_2 =
            context_->d_leveled->operator[](input1.depth_) % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(
                temp3_rotation, galois_key.device_location_[galois_elt].data(),
                temp4_rotation, context_->modulus_->data(), first_rns_mod_count,
                current_decomp_count, current_rns_mod_count, iteration_count_1,
                iteration_count_2, input1.depth_, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp3_rotation, key_location.data(), temp4_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<dim3((context_->n >> 8),
                                                      current_decomp_count, 2),
                                                 256, 0, stream>>>(
            temp4_rotation, temp0_rotation, output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galois_elt, context_->n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));

        output.scheme_ = context_->scheme_;
        output.context_id_ = context_->context_id_;
        output.ring_size_ = context_->n;
        output.coeff_modulus_count_ = context_->Q_size;
        output.cipher_size_ = 2;
        output.depth_ = input1.depth_;
        output.scale_ = input1.scale_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.encoding_ = input1.encoding_;
        output.rescale_required_ = input1.rescale_required_;
        output.relinearization_required_ = input1.relinearization_required_;
        output.ciphertext_generated_ = true;
    }

    __host__ void HEOperator<Scheme::CKKS>::switchkey_ckks_method_I(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Switchkey<Scheme::CKKS>& switch_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation +
            (context_->n * context_->Q_size * context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        cipher_broadcast_switchkey_leveled_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(temp0_rotation, temp1_rotation, temp2_rotation,
                      context_->modulus_->data(), context_->n_power,
                      first_rns_mod_count, current_rns_mod_count,
                      current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        int iteration_count_1 = current_decomp_count / 4;
        int iteration_count_2 = current_decomp_count % 4;
        if (switch_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp2_rotation, switch_key.data(), temp3_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp2_rotation, key_location.data(), temp3_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt2 = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data() + first_decomp_count,
            .stream = stream};

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            temp3_rotation,
            context_->intt_table_->data() +
                (first_decomp_count << context_->n_power),
            context_->modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<
            dim3((context_->n >> 8), 2, 1), 256, 0, stream>>>(
            temp3_rotation, temp2_rotation, context_->modulus_->data(),
            context_->half_p_->data(), context_->half_mod_->data(),
            context_->n_power, first_decomp_count, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp2_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        // TODO: Merge with previous one
        gpuntt::GPU_NTT_Inplace(temp1_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                current_decomp_count, current_decomp_count);

        divide_round_lastq_leveled_stage_two_switchkey_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(temp2_rotation, temp3_rotation, temp1_rotation,
                      output_memory.data(), context_->modulus_->data(),
                      context_->last_q_modinv_->data(), context_->n_power,
                      current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::switchkey_ckks_method_II(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Switchkey<Scheme::CKKS>& switch_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size) +
                (2 * context_->n * context_->d_leveled->operator[](0) *
                 context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation + (context_->n * context_->Q_size);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * context_->n * context_->d_leveled->operator[](0) *
             context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        cipher_broadcast_switchkey_method_II_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(temp0_rotation, temp1_rotation, temp2_rotation,
                      context_->modulus_->data(), context_->n_power,
                      current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((context_->n >> 8),
                 context_->d_leveled->operator[](input1.depth_), 1),
            256, 0, stream>>>(
            temp2_rotation, temp3_rotation, context_->modulus_->data(),
            context_->base_change_matrix_D_to_Qtilda_leveled
                ->
                operator[](input1.depth_)
                .data(),
            context_->Mi_inv_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->prod_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->I_j_leveled->operator[](input1.depth_).data(),
            context_->I_location_leveled->operator[](input1.depth_).data(),
            context_->n_power, context_->d_leveled->operator[](input1.depth_),
            current_rns_mod_count, current_decomp_count, input1.depth_,
            context_->prime_location_leveled->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            context_->d_leveled->operator[](input1.depth_) *
                current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        int iteration_count_1 =
            context_->d_leveled->operator[](input1.depth_) / 4;
        int iteration_count_2 =
            context_->d_leveled->operator[](input1.depth_) % 4;
        if (switch_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp3_rotation, switch_key.data(), temp4_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp3_rotation, key_location.data(), temp4_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        divide_round_lastq_extended_leveled_kernel<<<
            dim3((context_->n >> 8), current_decomp_count, 2), 256, 0,
            stream>>>(
            temp4_rotation, temp3_rotation, context_->modulus_->data(),
            context_->half_p_->data(), context_->half_mod_->data(),
            context_->last_q_modinv_->data(), context_->n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp3_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        // TODO: Fused the redundant kernels
        // TODO: Merge with previous one
        gpuntt::GPU_NTT_Inplace(temp1_rotation, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                current_decomp_count, current_decomp_count);

        addition_switchkey<<<dim3((context_->n >> 8), current_decomp_count, 2),
                             256, 0, stream>>>(
            temp3_rotation, temp1_rotation, output_memory.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::conjugate_ckks_method_I(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& conjugate_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        int galois_elt = conjugate_key.galois_elt_zero;

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size * context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation +
            (context_->n * context_->Q_size * context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        // TODO: make it efficient
        ckks_duplicate_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                     1),
                                256, 0, stream>>>(
            temp0_rotation, temp2_rotation, context_->modulus_->data(),
            context_->n_power, first_rns_mod_count, current_rns_mod_count,
            current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = current_decomp_count / 4;
        int iteration_count_2 = current_decomp_count % 4;
        if (conjugate_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp2_rotation, conjugate_key.c_data(),
                          temp3_rotation, context_->modulus_->data(),
                          first_rns_mod_count, current_decomp_count,
                          iteration_count_1, iteration_count_2,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(conjugate_key.zero_host_location_,
                                              stream);
            keyswitch_multiply_accumulate_leveled_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp2_rotation, key_location.data(), temp3_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, iteration_count_1,
                          iteration_count_2, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<dim3((context_->n >> 8),
                                                      current_decomp_count, 2),
                                                 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galois_elt, context_->n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::CKKS>::conjugate_ckks_method_II(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        Galoiskey<Scheme::CKKS>& conjugate_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = context_->Q_prime_size;
        int current_rns_mod_count = context_->Q_prime_size - input1.depth_;

        int first_decomp_count = context_->Q_size;
        int current_decomp_count = context_->Q_size - input1.depth_;

        DeviceVector<Data64> output_memory(
            (2 * context_->n * current_decomp_count), stream);

        int galois_elt = conjugate_key.galois_elt_zero;

        DeviceVector<Data64> temp_rotation(
            (2 * context_->n * context_->Q_size) +
                (2 * context_->n * context_->Q_size) +
                (context_->n * context_->Q_size) +
                (2 * context_->n * context_->d_leveled->operator[](0) *
                 context_->Q_prime_size) +
                (2 * context_->n * context_->Q_prime_size),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation =
            temp0_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp2_rotation =
            temp1_rotation + (2 * context_->n * context_->Q_size);
        Data64* temp3_rotation =
            temp2_rotation + (context_->n * context_->Q_size);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * context_->n * context_->d_leveled->operator[](0) *
             context_->Q_prime_size);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(input1.data(), temp0_rotation,
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         2 * current_decomp_count, current_decomp_count);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((context_->n >> 8),
                 context_->d_leveled->operator[](input1.depth_), 1),
            256, 0, stream>>>(
            temp0_rotation + (current_decomp_count << context_->n_power),
            temp3_rotation, context_->modulus_->data(),
            context_->base_change_matrix_D_to_Qtilda_leveled
                ->
                operator[](input1.depth_)
                .data(),
            context_->Mi_inv_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->prod_D_to_Qtilda_leveled->operator[](input1.depth_)
                .data(),
            context_->I_j_leveled->operator[](input1.depth_).data(),
            context_->I_location_leveled->operator[](input1.depth_).data(),
            context_->n_power, context_->d_leveled->operator[](input1.depth_),
            current_rns_mod_count, current_decomp_count, input1.depth_,
            context_->prime_location_leveled->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, context_->ntt_table_->data(),
            context_->modulus_->data(), cfg_ntt,
            context_->d_leveled->operator[](input1.depth_) *
                current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 =
            context_->d_leveled->operator[](input1.depth_) / 4;
        int iteration_count_2 =
            context_->d_leveled->operator[](input1.depth_) % 4;
        if (conjugate_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp3_rotation, conjugate_key.c_data(),
                          temp4_rotation, context_->modulus_->data(),
                          first_rns_mod_count, current_decomp_count,
                          current_rns_mod_count, iteration_count_1,
                          iteration_count_2, input1.depth_, context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(conjugate_key.zero_host_location_,
                                              stream);
            keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                dim3((context_->n >> 8), current_rns_mod_count, 1), 256, 0,
                stream>>>(temp3_rotation, key_location.data(), temp4_rotation,
                          context_->modulus_->data(), first_rns_mod_count,
                          current_decomp_count, current_rns_mod_count,
                          iteration_count_1, iteration_count_2, input1.depth_,
                          context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, context_->intt_table_->data(),
            context_->modulus_->data(), cfg_intt, 2 * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<dim3((context_->n >> 8),
                                                      current_decomp_count, 2),
                                                 256, 0, stream>>>(
            temp4_rotation, temp0_rotation, output_memory.data(),
            context_->modulus_->data(), context_->half_p_->data(),
            context_->half_mod_->data(), context_->last_q_modinv_->data(),
            galois_elt, context_->n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            context_->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    //                       BOOTSRAPPING                         //
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    __host__ Plaintext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::operator_plaintext(cudaStream_t stream)
    {
        Plaintext<Scheme::CKKS> plain;

        plain.scheme_ = context_->scheme_;
        plain.context_id_ = context_->context_id_;
        plain.plain_size_ = context_->n * context_->Q_size; // context_->n
        plain.depth_ = 0;
        plain.scale_ = 0;
        plain.in_ntt_domain_ = true;
        plain.encoding_ = encoding::SLOT;

        plain.device_locations_ =
            DeviceVector<Data64>(plain.plain_size_, stream);

        return plain;
    }

    __host__ Plaintext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::operator_from_plaintext(
        Plaintext<Scheme::CKKS>& input, cudaStream_t stream)
    {
        Plaintext<Scheme::CKKS> plain;

        plain.scheme_ = input.scheme_;
        plain.context_id_ = input.context_id_;
        plain.plain_size_ = input.plain_size_;
        plain.depth_ = input.depth_;
        plain.scale_ = input.scale_;
        plain.in_ntt_domain_ = input.in_ntt_domain_;
        plain.encoding_ = input.encoding_;

        plain.storage_type_ = storage_type::DEVICE;
        plain.device_locations_ =
            DeviceVector<Data64>(plain.plain_size_, stream);

        return plain;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::operator_ciphertext(double scale,
                                                  cudaStream_t stream)
    {
        Ciphertext<Scheme::CKKS> cipher;

        cipher.coeff_modulus_count_ = context_->Q_size;
        cipher.cipher_size_ = 2; // default
        cipher.ring_size_ = context_->n; // context_->n
        cipher.depth_ = 0;

        cipher.scheme_ = context_->scheme_;
        cipher.context_id_ = context_->context_id_;
        cipher.in_ntt_domain_ = true;
        cipher.encoding_ = encoding::SLOT;
        cipher.storage_type_ = storage_type::DEVICE;

        cipher.rescale_required_ = false;
        cipher.relinearization_required_ = false;
        cipher.scale_ = scale;
        cipher.ciphertext_generated_ = true;

        int cipher_memory_size =
            2 * (context_->Q_size - cipher.depth_) * context_->n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::operator_from_ciphertext(
        Ciphertext<Scheme::CKKS>& input, cudaStream_t stream)
    {
        Ciphertext<Scheme::CKKS> cipher;

        cipher.coeff_modulus_count_ = input.coeff_modulus_count_;
        cipher.cipher_size_ = input.cipher_size_;
        cipher.ring_size_ = input.ring_size_;
        cipher.depth_ = input.depth_;

        cipher.scheme_ = input.scheme_;
        cipher.context_id_ = input.context_id_;
        cipher.in_ntt_domain_ = input.in_ntt_domain_;
        cipher.encoding_ = input.encoding_;

        cipher.storage_type_ = storage_type::DEVICE;

        cipher.rescale_required_ = input.rescale_required_;
        cipher.relinearization_required_ = input.relinearization_required_;
        cipher.scale_ = input.scale_;
        cipher.ciphertext_generated_ = true;

        int cipher_memory_size =
            2 * (context_->Q_size - cipher.depth_) * context_->n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ std::vector<int>
    HEOperator<Scheme::CKKS>::rotation_index_generator(uint64_t n, int K, int M)
    {
        if (K <= 0 || M < 0 || M >= K)
            return {};

        auto balance_mod_2k = [](uint64_t n, int K)
        {
            if (K < 64)
            {
                uint64_t mod = (1ULL << K);
                uint64_t mask = mod - 1ULL;
                uint64_t m = n & mask;
                int64_t half = static_cast<int64_t>(mod >> 1);
                return (m > half)
                           ? static_cast<int64_t>(m) - static_cast<int64_t>(mod)
                           : static_cast<int64_t>(m);
            }
            return static_cast<int64_t>(n);
        };

        int64_t x = balance_mod_2k(n, K);

        std::vector<long long> coeff(M + 1, 0);

        // NAF + fold
        for (int i = 0; i < K && x != 0; i++)
        {
            if (x & 1LL)
            {
                int ui = 1 - static_cast<int>((x >> 1) & 1) * 2; // ±1
                if (i <= M)
                    coeff[i] += ui;
                else
                    coeff[M] += static_cast<long long>(ui) << (i - M);
                x = (x - ui) >> 1;
            }
            else
            {
                x >>= 1;
            }
        }

        // normalize
        for (int i = 0; i < M; i++)
        {
            long long v = coeff[i];
            if (v >= 2 || v <= -2)
            {
                long long carry = v / 2;
                coeff[i] -= carry * 2;
                coeff[i + 1] += carry;
            }
        }

        // smoothing
        for (int i = M; i >= 1; i--)
        {
            long long up = coeff[i];
            long long low = coeff[i - 1];
            if (up > 0 && low < 0)
            {
                long long up2 = up - 1, low2 = low + 2;
                if (std::llabs(up2) + std::llabs(low2) <=
                    std::llabs(up) + std::llabs(low))
                {
                    coeff[i] = up2;
                    coeff[i - 1] = low2;
                }
            }
            else if (up < 0 && low > 0)
            {
                long long up2 = up + 1, low2 = low - 2;
                if (std::llabs(up2) + std::llabs(low2) <=
                    std::llabs(up) + std::llabs(low))
                {
                    coeff[i] = up2;
                    coeff[i - 1] = low2;
                }
            }
        }

        // coeff → list
        std::vector<int> out;
        for (int i = 0; i <= M; i++)
        {
            long long c = coeff[i];
            if (!c)
                continue;
            int term = static_cast<int>(1ULL << i);
            if (c > 0)
                while (c--)
                    out.push_back(term);
            else
                while (c++)
                    out.push_back(-term);
        }
        return out;
    }

    __host__ void HEOperator<Scheme::CKKS>::quick_ckks_encoder_vec_complex(
        Complex64* input, Data64* output, const double scale, int rns_count)
    {
        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = 0;

        gpufft::GPU_Special_FFT(input, special_ifft_roots_table_->data(),
                                cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            output, input, context_->modulus_->data(), rns_count, two_pow_64_,
            reverse_order_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(output, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt, rns_count,
                                rns_count);
    }

    __host__ void HEOperator<Scheme::CKKS>::quick_ckks_encoder_constant_complex(
        Complex64 input, Data64* output, const double scale)
    {
        // std::vector<Complex64> in = {input};
        std::vector<Complex64> in;
        for (int i = 0; i < slot_count_; i++)
        {
            in.push_back(input);
        }
        DeviceVector<Complex64> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), in.data(), in.size() * sizeof(Complex64),
                   cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = 0;

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            output, message_gpu.data(), context_->modulus_->data(),
            context_->Q_size, two_pow_64_, reverse_order_->data(),
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(output, context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);
    }

    __host__ void HEOperator<Scheme::CKKS>::quick_ckks_encoder_constant_double(
        double input, Data64* output, const double scale)
    {
        double value = input * scale;

        encode_kernel_double_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                               256>>>(
            output, value, context_->modulus_->data(), context_->Q_size,
            two_pow_64_, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::CKKS>::quick_ckks_encoder_constant_integer(
        std::int64_t input, Data64* output, const double scale)
    {
        double value = static_cast<double>(input) * scale;

        encode_kernel_double_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                               256>>>(
            output, value, context_->modulus_->data(), context_->Q_size,
            two_pow_64_, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ std::vector<heongpu::DeviceVector<Data64>>
    HEOperator<Scheme::CKKS>::encode_V_matrixs(Vandermonde& vandermonde,
                                               int start_level)
    {
        if (context_->keyswitching_type_ !=
            keyswitching_type::KEYSWITCHING_METHOD_II)
        {
            throw std::invalid_argument(
                "CoeffToSlot/SlotToCoeff require KEYSWITCHING_METHOD_II "
                "(more than one P prime)!");
        }

        std::vector<heongpu::DeviceVector<Data64>> result;

        int rns_count_base = start_level + 2 - vandermonde.StoC_piece_;
        int n = context_->n;
        int Q_size = context_->Q_size;
        int P_size = context_->P_size;
        int Q_prime_size = context_->Q_prime_size;

        for (int m = 0; m < vandermonde.StoC_piece_; m++)
        {
            int current_decomp_count =
                rns_count_base + m; // cdc = number of Q limbs at this level
            int pql_count = current_decomp_count + P_size; // PQ_l limb count

            heongpu::DeviceVector<Data64> temp_encoded(
                (vandermonde.V_matrixs_index_[m].size() * pql_count)
                << (vandermonde.log_num_slots_ + 1));

            double scale = static_cast<double>(
                context_->prime_vector_[current_decomp_count - 1].value);

            // Temporary buffer for full Q_prime_size encoding
            heongpu::DeviceVector<Data64> temp_full(
                Q_prime_size << (vandermonde.log_num_slots_ + 1));

            for (int i = 0; i < vandermonde.V_matrixs_index_[m].size(); i++)
            {
                int matrix_location = (i << vandermonde.log_num_slots_);
                int plaintext_location =
                    ((i * pql_count) << (vandermonde.log_num_slots_ + 1));

                // Encode with all Q_prime_size primes
                quick_ckks_encoder_vec_complex(
                    vandermonde.V_matrixs_rotated_[m].data() + matrix_location,
                    temp_full.data(), scale, Q_prime_size);

                // Rearrange to PQ_l layout: [Q_0..Q_{cdc-1} |
                // P_0..P_{P_size-1}] Copy Q limbs (first current_decomp_count
                // limbs)
                cudaMemcpy(temp_encoded.data() + plaintext_location,
                           temp_full.data(),
                           current_decomp_count * n * sizeof(Data64),
                           cudaMemcpyDeviceToDevice);
                // Copy P limbs (from index Q_size in full encoding)
                cudaMemcpy(temp_encoded.data() + plaintext_location +
                               (current_decomp_count * n),
                           temp_full.data() + (Q_size * n),
                           P_size * n * sizeof(Data64),
                           cudaMemcpyDeviceToDevice);
            }

            result.push_back(std::move(temp_encoded));
        }

        return result;
    }

    __host__ std::vector<heongpu::DeviceVector<Data64>>
    HEOperator<Scheme::CKKS>::encode_V_inv_matrixs(Vandermonde& vandermonde,
                                                   int start_level)
    {
        if (context_->keyswitching_type_ !=
            keyswitching_type::KEYSWITCHING_METHOD_II)
        {
            throw std::invalid_argument(
                "CoeffToSlot/SlotToCoeff require KEYSWITCHING_METHOD_II "
                "(more than one P prime)!");
        }

        std::vector<heongpu::DeviceVector<Data64>> result;

        int rns_count_base = start_level + 2 - vandermonde.CtoS_piece_;
        int n = context_->n;
        int Q_size = context_->Q_size;
        int P_size = context_->P_size;
        int Q_prime_size = context_->Q_prime_size;

        for (int m = 0; m < vandermonde.CtoS_piece_; m++)
        {
            int current_decomp_count =
                rns_count_base + m; // cdc = number of Q limbs at this level
            int pql_count = current_decomp_count + P_size; // PQ_l limb count

            heongpu::DeviceVector<Data64> temp_encoded(
                (vandermonde.V_inv_matrixs_index_[m].size() * pql_count)
                << (vandermonde.log_num_slots_ + 1));

            double scale = static_cast<double>(
                context_->prime_vector_[current_decomp_count - 1].value);

            // Temporary buffer for full Q_prime_size encoding
            heongpu::DeviceVector<Data64> temp_full(
                Q_prime_size << (vandermonde.log_num_slots_ + 1));

            for (int i = 0; i < vandermonde.V_inv_matrixs_index_[m].size(); i++)
            {
                int matrix_location = (i << vandermonde.log_num_slots_);
                int plaintext_location =
                    ((i * pql_count) << (vandermonde.log_num_slots_ + 1));

                // Encode with all Q_prime_size primes
                quick_ckks_encoder_vec_complex(
                    vandermonde.V_inv_matrixs_rotated_[m].data() +
                        matrix_location,
                    temp_full.data(), scale, Q_prime_size);

                // Rearrange to PQ_l layout: [Q_0..Q_{cdc-1} |
                // P_0..P_{P_size-1}] Copy Q limbs (first current_decomp_count
                // limbs)
                cudaMemcpy(temp_encoded.data() + plaintext_location,
                           temp_full.data(),
                           current_decomp_count * n * sizeof(Data64),
                           cudaMemcpyDeviceToDevice);
                // Copy P limbs (from index Q_size in full encoding)
                cudaMemcpy(temp_encoded.data() + plaintext_location +
                               (current_decomp_count * n),
                           temp_full.data() + (Q_size * n),
                           P_size * n * sizeof(Data64),
                           cudaMemcpyDeviceToDevice);
            }

            result.push_back(std::move(temp_encoded));
        }

        return result;
    }

    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::multiply_matrix(
        Ciphertext<Scheme::CKKS>& cipher,
        std::vector<heongpu::DeviceVector<Data64>>& matrix,
        std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
        std::vector<std::vector<int>>& diags_matrices_bsgs_rot_n1_,
        std::vector<std::vector<int>>& diags_matrices_bsgs_rot_n2_,
        Galoiskey<Scheme::CKKS>& galois_key, const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher.stream();
        cipher.switch_stream(options.stream_);
        Ciphertext<Scheme::CKKS> result;
        result = cipher;
        cipher.switch_stream(old_stream);

        int matrix_count = diags_matrices_bsgs_.size();
        int n = context_->n;
        int Q_size = context_->Q_size;
        int P_size = context_->P_size;
        int Q_prime_size = context_->Q_prime_size;
        int first_rns_mod_count = Q_prime_size;
        cudaStream_t stream = options.stream_;

        for (int m = (matrix_count - 1); -1 < m; m--)
        {
            int current_level = result.depth_;
            int current_decomp_count = Q_size - current_level;
            int current_rns_mod_count = Q_prime_size - current_level;
            int pql_count = current_decomp_count + P_size;

            std::sort(diags_matrices_bsgs_rot_n2_[m].begin(),
                      diags_matrices_bsgs_rot_n2_[m].end());

            int n1 = diags_matrices_bsgs_rot_n2_[m].size();
            int n2 = diags_matrices_bsgs_[m].size();

            // ============================================================
            // Build PQ_l modulus array on host, upload to device
            // ============================================================
            std::vector<Modulus64> pq_mod_host(pql_count);
            for (int j = 0; j < current_decomp_count; j++)
                pq_mod_host[j] = context_->prime_vector_[j];
            for (int k = 0; k < P_size; k++)
                pq_mod_host[current_decomp_count + k] =
                    context_->prime_vector_[Q_size + k];
            DeviceVector<Modulus64> pq_modulus_dev(pq_mod_host);

            // Compute P mod q_j for each active Q prime
            std::vector<Data64> P_mod_q_host(current_decomp_count);
            for (int j = 0; j < current_decomp_count; j++)
            {
                // P = product of all P primes
                // Compute P mod q_j iteratively to avoid overflow
                uint64_t p_mod_qj = 1;
                uint64_t qj = context_->prime_vector_[j].value;
                for (int k = 0; k < P_size; k++)
                {
                    __uint128_t tmp =
                        (__uint128_t) p_mod_qj *
                        (context_->prime_vector_[Q_size + k].value % qj);
                    p_mod_qj = (uint64_t) (tmp % qj);
                }
                P_mod_q_host[j] = p_mod_qj;
            }
            DeviceVector<Data64> P_mod_q_dev(P_mod_q_host);

            // ============================================================
            // NTT location for PQ_l
            // ============================================================
            int counter_loc = first_rns_mod_count;
            int location = 0;
            for (int i = 0; i < current_level; i++)
            {
                location += counter_loc;
                counter_loc--;
            }

            // ============================================================
            // NTT/INTT configurations
            // ============================================================
            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = context_->n_power,
                .ntt_type = gpuntt::INVERSE,
                .ntt_layout = gpuntt::PerPolynomial,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = context_->n_inverse_->data(),
                .stream = stream};

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = context_->n_power,
                .ntt_type = gpuntt::FORWARD,
                .ntt_layout = gpuntt::PerPolynomial,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            // ============================================================
            // Hoisted Decompose of c1 (same as single hoisting)
            // ============================================================

            // temp0: INTT of ciphertext (c0_coeff || c1_coeff) in Q_l coeff
            DeviceVector<Data64> temp0(2 * n * Q_size, stream);
            {
                gpuntt::GPU_INTT(
                    result.data(), temp0.data(), context_->intt_table_->data(),
                    context_->modulus_->data(), cfg_intt,
                    2 * current_decomp_count, current_decomp_count);
            }

            // temp3: decomposed c1 in PQ_l NTT domain
            int d_level = context_->d_leveled->operator[](current_level);
            DeviceVector<Data64> temp3(2 * n * d_level * current_rns_mod_count,
                                       stream);

            {
                base_conversion_DtoQtilde_relin_leveled_kernel<<<
                    dim3((n >> 8), d_level, 1), 256, 0, stream>>>(
                    temp0.data() + (current_decomp_count << context_->n_power),
                    temp3.data(), context_->modulus_->data(),
                    context_->base_change_matrix_D_to_Qtilda_leveled
                        ->operator[](current_level)
                        .data(),
                    context_->Mi_inv_D_to_Qtilda_leveled
                        ->operator[](current_level)
                        .data(),
                    context_->prod_D_to_Qtilda_leveled
                        ->operator[](current_level)
                        .data(),
                    context_->I_j_leveled->operator[](current_level).data(),
                    context_->I_location_leveled->operator[](current_level)
                        .data(),
                    context_->n_power, d_level, current_rns_mod_count,
                    current_decomp_count, current_level,
                    context_->prime_location_leveled->data() + location);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            {
                gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                    temp3.data(), context_->ntt_table_->data(),
                    context_->modulus_->data(), cfg_ntt,
                    d_level * current_rns_mod_count, current_rns_mod_count,
                    new_prime_locations + location);
            }

            int iteration_count_1 = d_level / 4;
            int iteration_count_2 = d_level % 4;

            // ============================================================
            // Baby-Step Phase: produce n1 ciphertexts in PQ_l NTT domain
            // ============================================================

            // Each baby-step result: 2 components × pql_count limbs × n
            int baby_ct_size = 2 * pql_count * n;
            DeviceVector<Data64> baby_results(baby_ct_size * n1, stream);

            // Temporary for MultSum output (2 components ×
            // current_rns_mod_count limbs)
            DeviceVector<Data64> temp4(2 * n * current_rns_mod_count, stream);

            // Precompute P·c0 once — reused in every baby step (both the
            // zero-rotation case and the add-after-keyswitch case).
            DeviceVector<Data64> Pc0(pql_count * n, stream);
            {
                broadcast_scale_P_kernel<<<dim3((n >> 8), pql_count, 1), 256, 0,
                                           stream>>>(
                    result.data(), Pc0.data(), P_mod_q_dev.data(),
                    pq_modulus_dev.data(), context_->n_power,
                    current_decomp_count, pql_count);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            for (int i = 0; i < n1; i++)
            {
                int baby_offset = baby_ct_size * i;

                if (diags_matrices_bsgs_rot_n2_[m][i] == 0)
                {
                    // Zero rotation: baby_result = (P·c0, P·c1) in PQ_l NTT
                    // Component 0: copy precomputed P·c0
                    cudaMemcpyAsync(baby_results.data() + baby_offset,
                                    Pc0.data(), pql_count * n * sizeof(Data64),
                                    cudaMemcpyDeviceToDevice, stream);

                    // Component 1: P·c1
                    {
                        broadcast_scale_P_kernel<<<dim3((n >> 8), pql_count, 1),
                                                   256, 0, stream>>>(
                            result.data() + (current_decomp_count * n),
                            baby_results.data() + baby_offset + (pql_count * n),
                            P_mod_q_dev.data(), pq_modulus_dev.data(),
                            context_->n_power, current_decomp_count, pql_count);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }

                    continue;
                }

                int shift_n1 = diags_matrices_bsgs_rot_n2_[m][i];
                int galois_elt =
                    steps_to_galois_elt(shift_n1, n, galois_key.group_order_);

                // MultSum: decomposed c1 × rotation key -> (ks0, ks1) in PQ_l
                // NTT
                {
                    if (galois_key.storage_type_ == storage_type::DEVICE)
                    {
                        keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                            dim3((n >> 8), current_rns_mod_count, 1), 256, 0,
                            stream>>>(
                            temp3.data(),
                            galois_key.device_location_[galois_elt].data(),
                            temp4.data(), context_->modulus_->data(),
                            first_rns_mod_count, current_decomp_count,
                            current_rns_mod_count, iteration_count_1,
                            iteration_count_2, current_level,
                            context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                    else
                    {
                        DeviceVector<Data64> key_location(
                            galois_key.host_location_[galois_elt], stream);
                        keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                            dim3((n >> 8), current_rns_mod_count, 1), 256, 0,
                            stream>>>(temp3.data(), key_location.data(),
                                      temp4.data(), context_->modulus_->data(),
                                      first_rns_mod_count, current_decomp_count,
                                      current_rns_mod_count, iteration_count_1,
                                      iteration_count_2, current_level,
                                      context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                // Add precomputed P·c0 to component 0 of keyswitch output
                {
                    addition_pql_kernel<<<dim3((n >> 8), pql_count, 1), 256, 0,
                                          stream>>>(
                        temp4.data(), Pc0.data(), temp4.data(),
                        pq_modulus_dev.data(), context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                // NTT-domain Galois permutation: avoid INTT->permute->NTT
                {
                    galois_permute_ntt_pql_kernel<<<
                        dim3((n >> 8), pql_count, 2), 256, 0, stream>>>(
                        temp4.data(), baby_results.data() + baby_offset,
                        galois_elt, context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // ============================================================
            // Giant-Step Phase
            // ============================================================
            // Accumulator in PQ_l NTT domain (2 components × pql_count limbs)
            DeviceVector<Data64> gs_accum(2 * pql_count * n, stream);
            cudaMemsetAsync(gs_accum.data(), 0,
                            2 * pql_count * n * sizeof(Data64), stream);

            // Temp buffers for giant-step operations
            DeviceVector<Data64> u_pql(2 * pql_count * n, stream);
            DeviceVector<Data64> u1_Q(current_decomp_count * n, stream);
            DeviceVector<Data64> temp3_gs(
                2 * n * d_level * current_rns_mod_count, stream);
            DeviceVector<Data64> temp4_gs(2 * n * current_rns_mod_count,
                                          stream);
            DeviceVector<Data64> permuted_gs(2 * pql_count * n, stream);

            int counter = 0;
            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                int real_shift = diags_matrices_bsgs_rot_n1_[m][j];

                // Build index table mapping each iteration to its baby-step
                // position in baby_results, avoiding the gather copy.
                int inner_n1 = diags_matrices_bsgs_[m][j].size();
                std::vector<int> ct_indices_host(inner_n1);
                for (int k = 0; k < inner_n1; k++)
                {
                    int real_n2 = diags_matrices_bsgs_[m][j][k] - real_shift;
                    auto it = std::find(diags_matrices_bsgs_rot_n2_[m].begin(),
                                        diags_matrices_bsgs_rot_n2_[m].end(),
                                        real_n2);
                    ct_indices_host[k] = static_cast<int>(std::distance(
                        diags_matrices_bsgs_rot_n2_[m].begin(), it));
                }
                DeviceVector<int> ct_indices_dev(ct_indices_host, stream);

                // Inner product: baby_results[ct_indices[i]] × plaintext diags
                // -> u
                {
                    int matrix_plaintext_location = (counter * pql_count)
                                                    << context_->n_power;
                    cipherplain_multiply_accumulate_indexed_kernel<<<
                        dim3((n >> 8), pql_count, 2), 256, 0, stream>>>(
                        baby_results.data(),
                        matrix[m].data() + matrix_plaintext_location,
                        u_pql.data(), pq_modulus_dev.data(),
                        ct_indices_dev.data(), inner_n1, pql_count, pql_count,
                        context_->n_power);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                counter += inner_n1;

                if (real_shift == 0)
                {
                    // No giant-step rotation needed, add u directly to
                    // accumulator
                    addition_pql_kernel<<<dim3((n >> 8), pql_count, 2), 256, 0,
                                          stream>>>(
                        gs_accum.data(), u_pql.data(), gs_accum.data(),
                        pq_modulus_dev.data(), context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                    continue;
                }

                int galois_elt_gs =
                    steps_to_galois_elt(real_shift, n, galois_key.group_order_);

                // ModDown u1 (component 1) from PQ_l -> Q_l
                // First INTT u1 from PQ_l NTT -> PQ_l coeff
                // u1 is at u_pql + pql_count * n
                {
                    gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                        u_pql.data() + (pql_count * n),
                        context_->intt_table_->data(),
                        context_->modulus_->data(), cfg_intt, pql_count,
                        pql_count, new_prime_locations + location);
                }

                // ModDown u1: PQ_l coeff -> Q_l coeff
                // Input: u1 in PQ_l coeff (pql_count limbs)
                // Output: u1_Q in Q_l coeff (current_decomp_count limbs)
                {
                    divide_round_lastq_extended_leveled_kernel<<<
                        dim3((n >> 8), current_decomp_count, 1), 256, 0,
                        stream>>>(
                        u_pql.data() + (pql_count * n), u1_Q.data(),
                        context_->modulus_->data(), context_->half_p_->data(),
                        context_->half_mod_->data(),
                        context_->last_q_modinv_->data(), context_->n_power,
                        pql_count, current_decomp_count, first_rns_mod_count,
                        Q_size, P_size);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                // Decompose u1_Q -> PQ_l coeff, then NTT
                {
                    base_conversion_DtoQtilde_relin_leveled_kernel<<<
                        dim3((n >> 8), d_level, 1), 256, 0, stream>>>(
                        u1_Q.data(), temp3_gs.data(),
                        context_->modulus_->data(),
                        context_->base_change_matrix_D_to_Qtilda_leveled
                            ->operator[](current_level)
                            .data(),
                        context_->Mi_inv_D_to_Qtilda_leveled
                            ->operator[](current_level)
                            .data(),
                        context_->prod_D_to_Qtilda_leveled
                            ->operator[](current_level)
                            .data(),
                        context_->I_j_leveled->operator[](current_level).data(),
                        context_->I_location_leveled->operator[](current_level)
                            .data(),
                        context_->n_power, d_level, current_rns_mod_count,
                        current_decomp_count, current_level,
                        context_->prime_location_leveled->data() + location);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                {
                    gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                        temp3_gs.data(), context_->ntt_table_->data(),
                        context_->modulus_->data(), cfg_ntt,
                        d_level * current_rns_mod_count, current_rns_mod_count,
                        new_prime_locations + location);
                }

                // MultSum: decomposed u1 × giant-step key -> (gs0, gs1) in PQ_l
                // NTT
                {
                    if (galois_key.storage_type_ == storage_type::DEVICE)
                    {
                        keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                            dim3((n >> 8), current_rns_mod_count, 1), 256, 0,
                            stream>>>(
                            temp3_gs.data(),
                            galois_key.device_location_[galois_elt_gs].data(),
                            temp4_gs.data(), context_->modulus_->data(),
                            first_rns_mod_count, current_decomp_count,
                            current_rns_mod_count, iteration_count_1,
                            iteration_count_2, current_level,
                            context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                    else
                    {
                        DeviceVector<Data64> key_location(
                            galois_key.host_location_[galois_elt_gs], stream);
                        keyswitch_multiply_accumulate_leveled_method_II_kernel<<<
                            dim3((n >> 8), current_rns_mod_count, 1), 256, 0,
                            stream>>>(temp3_gs.data(), key_location.data(),
                                      temp4_gs.data(),
                                      context_->modulus_->data(),
                                      first_rns_mod_count, current_decomp_count,
                                      current_rns_mod_count, iteration_count_1,
                                      iteration_count_2, current_level,
                                      context_->n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                // Add u0 to component 0 of temp4_gs directly in NTT domain
                {
                    addition_pql_kernel<<<dim3((n >> 8), pql_count, 1), 256, 0,
                                          stream>>>(
                        temp4_gs.data(), u_pql.data(), temp4_gs.data(),
                        pq_modulus_dev.data(), context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                // NTT-domain Galois permutation: avoid INTT->permute->NTT
                {
                    galois_permute_ntt_pql_kernel<<<
                        dim3((n >> 8), pql_count, 2), 256, 0, stream>>>(
                        temp4_gs.data(), permuted_gs.data(), galois_elt_gs,
                        context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                // Accumulate
                {
                    addition_pql_kernel<<<dim3((n >> 8), pql_count, 2), 256, 0,
                                          stream>>>(
                        gs_accum.data(), permuted_gs.data(), gs_accum.data(),
                        pq_modulus_dev.data(), context_->n_power, pql_count);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // ============================================================
            // Final ModDown: PQ_l -> Q_l
            // ============================================================
            // INTT accumulator from PQ_l NTT -> PQ_l coeff
            {
                gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                    gs_accum.data(), context_->intt_table_->data(),
                    context_->modulus_->data(), cfg_intt, 2 * pql_count,
                    pql_count, new_prime_locations + location);
            }

            // ModDown both components: PQ_l coeff -> Q_l coeff
            DeviceVector<Data64> final_ct(2 * current_decomp_count * n, stream);
            {
                divide_round_lastq_extended_leveled_kernel<<<
                    dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
                    gs_accum.data(), final_ct.data(),
                    context_->modulus_->data(), context_->half_p_->data(),
                    context_->half_mod_->data(),
                    context_->last_q_modinv_->data(), context_->n_power,
                    pql_count, current_decomp_count, first_rns_mod_count,
                    Q_size, P_size);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // NTT final_ct -> Q_l NTT domain
            {
                gpuntt::GPU_NTT_Inplace(
                    final_ct.data(), context_->ntt_table_->data(),
                    context_->modulus_->data(), cfg_ntt,
                    2 * current_decomp_count, current_decomp_count);
            }

            // Copy into result ciphertext
            result.device_locations_ = std::move(final_ct);
            result.scheme_ = context_->scheme_;
            result.context_id_ = context_->context_id_;
            result.ring_size_ = n;
            result.coeff_modulus_count_ = Q_size;
            result.cipher_size_ = 2;
            result.depth_ = current_level;
            result.in_ntt_domain_ = true;
            result.rescale_required_ = false;
            result.relinearization_required_ = false;
            result.ciphertext_generated_ = true;

            // Rescale
            double scale = static_cast<double>(
                context_->prime_vector_[current_decomp_count-1].value);
            result.scale_ = result.scale_ * scale;
            result.rescale_required_ = true;
            rescale_inplace(result, options);
        }

        return result;
    }

    __host__ std::vector<Ciphertext<Scheme::CKKS>>
    HEOperator<Scheme::CKKS>::coeff_to_slot(
        Ciphertext<Scheme::CKKS>& cipher, Galoiskey<Scheme::CKKS>& galois_key,
        CKKSEncodingTransformContext& transform_context,
        const ExecutionOptions& options)
    {
        cipher.validate_context(context_);

        if (!transform_context.generated_)
        {
            throw std::invalid_argument(
                "Encoding transform context is not generated!");
        }

        if (cipher.encoding_ != encoding::COEFFICIENT)
        {
            throw std::invalid_argument(
                "coeff_to_slot expects coefficient-encoded ciphertext.");
        }

        if (cipher.depth_ != transform_context.CtoS_level_)
        {
            throw std::logic_error("coeff_to_slot input level does not match "
                                   "transform_context.CtoS_start_level");
        }

        Ciphertext<Scheme::CKKS> c1 = multiply_matrix(
            cipher, transform_context.V_inv_matrixs_rotated_encoded_,
            transform_context.diags_matrices_inv_bsgs_,
            transform_context.diags_matrices_inv_bsgs_rot_n1_,
            transform_context.diags_matrices_inv_bsgs_rot_n2_, galois_key,
            options);

        Ciphertext<Scheme::CKKS> c2 = operator_ciphertext(0, options.stream_);
        conjugate(c1, c2, galois_key, options);

        Ciphertext<Scheme::CKKS> result0 =
            operator_ciphertext(0, options.stream_);
        add(c1, c2, result0, options);

        Ciphertext<Scheme::CKKS> result1 =
            operator_ciphertext(0, options.stream_);
        sub(c1, c2, result1, options);

        int current_decomp_count = context_->Q_size - result1.depth_;
        cipher_div_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                      2),
                                 256, 0, options.stream_>>>(
            result1.data(), result1.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result0.encoding_ = encoding::SLOT;
        result1.encoding_ = encoding::SLOT;

        std::vector<Ciphertext<Scheme::CKKS>> result;
        result.push_back(std::move(result0));
        result.push_back(std::move(result1));

        return result;
    }

    __host__ std::vector<Ciphertext<Scheme::CKKS>>
    HEOperator<Scheme::CKKS>::coeff_to_slot(Ciphertext<Scheme::CKKS>& cipher,
                                            Galoiskey<Scheme::CKKS>& galois_key,
                                            const ExecutionOptions& options)
    {
        cipher.validate_context(context_);

        if (cipher.encoding_ != encoding::COEFFICIENT)
        {
            throw std::invalid_argument(
                "coeff_to_slot expects coefficient-encoded ciphertext.");
        }

        Ciphertext<Scheme::CKKS> c1 = multiply_matrix(
            cipher, V_inv_matrixs_rotated_encoded_, diags_matrices_inv_bsgs_,
            diags_matrices_inv_bsgs_rot_n1_, diags_matrices_inv_bsgs_rot_n2_,
            galois_key, options);

        Ciphertext<Scheme::CKKS> c2 = operator_ciphertext(0, options.stream_);

        conjugate(c1, c2, galois_key, options); // conjugate

        Ciphertext<Scheme::CKKS> result0 =
            operator_ciphertext(0, options.stream_);
        add(c1, c2, result0, options);

        Ciphertext<Scheme::CKKS> result1 =
            operator_ciphertext(0, options.stream_);
        sub(c1, c2, result1, options);

        int current_decomp_count = context_->Q_size - result1.depth_;
        cipher_div_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                      2),
                                 256, 0, options.stream_>>>(
            result1.data(), result1.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result0.encoding_ = encoding::SLOT;
        result1.encoding_ = encoding::SLOT;

        std::vector<Ciphertext<Scheme::CKKS>> result;
        result.push_back(std::move(result0));
        result.push_back(std::move(result1));

        return result;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::solo_coeff_to_slot(
        Ciphertext<Scheme::CKKS>& cipher, Galoiskey<Scheme::CKKS>& galois_key,
        const ExecutionOptions& options)
    {
        if (cipher.encoding_ != encoding::COEFFICIENT)
        {
            throw std::invalid_argument("solo_coeff_to_slot expects "
                                        "coefficient-encoded ciphertext.");
        }

        Ciphertext<Scheme::CKKS> c1 = multiply_matrix(
            cipher, V_inv_matrixs_rotated_encoded_, diags_matrices_inv_bsgs_,
            diags_matrices_inv_bsgs_rot_n1_, diags_matrices_inv_bsgs_rot_n2_,
            galois_key, options);

        Ciphertext<Scheme::CKKS> c2 = operator_ciphertext(0, options.stream_);
        conjugate(c1, c2, galois_key, options); // conjugate

        Ciphertext<Scheme::CKKS> result =
            operator_ciphertext(0, options.stream_);
        add(c1, c2, result, options);
        result.encoding_ = encoding::SLOT;

        return result;
    }

    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::slot_to_coeff(
        Ciphertext<Scheme::CKKS>& cipher0, Ciphertext<Scheme::CKKS>& cipher1,
        Galoiskey<Scheme::CKKS>& galois_key,
        CKKSEncodingTransformContext& transform_context,
        const ExecutionOptions& options)
    {
        cipher0.validate_context(context_);
        cipher1.validate_context(context_);

        if (!transform_context.generated_)
        {
            throw std::invalid_argument(
                "Encoding transform context is not generated!");
        }

        if ((cipher0.encoding_ != encoding::SLOT) ||
            (cipher1.encoding_ != encoding::SLOT))
        {
            throw std::invalid_argument(
                "slot_to_coeff expects slot-encoded ciphertext inputs.");
        }

        if (cipher0.depth_ != cipher1.depth_)
        {
            throw std::logic_error("slot_to_coeff expects equal levels for "
                                   "both slot ciphertexts.");
        }

        if (cipher0.depth_ != transform_context.StoC_level_)
        {
            throw std::logic_error("slot_to_coeff input level does not match "
                                   "transform_context.StoC_start_level");
        }

        cudaStream_t old_stream = cipher1.stream();
        cipher1.switch_stream(options.stream_);
        Ciphertext<Scheme::CKKS> result;
        result = cipher1;
        cipher1.switch_stream(old_stream);

        int current_decomp_count = context_->Q_size - cipher1.depth_;
        cipher_mult_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                       2),
                                  256, 0, options.stream_>>>(
            result.data(), result.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        add(result, cipher0, result, options);

        Ciphertext<Scheme::CKKS> c1 = multiply_matrix(
            result, transform_context.V_matrixs_rotated_encoded_,
            transform_context.diags_matrices_bsgs_,
            transform_context.diags_matrices_bsgs_rot_n1_,
            transform_context.diags_matrices_bsgs_rot_n2_, galois_key, options);
        c1.encoding_ = encoding::COEFFICIENT;

        return c1;
    }

    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::slot_to_coeff(
        Ciphertext<Scheme::CKKS>& cipher0, Ciphertext<Scheme::CKKS>& cipher1,
        Galoiskey<Scheme::CKKS>& galois_key, const ExecutionOptions& options)
    {
        cipher0.validate_context(context_);
        cipher1.validate_context(context_);

        if ((cipher0.encoding_ != encoding::SLOT) ||
            (cipher1.encoding_ != encoding::SLOT))
        {
            throw std::invalid_argument(
                "slot_to_coeff expects slot-encoded ciphertext inputs.");
        }

        cudaStream_t old_stream = cipher1.stream();
        cipher1.switch_stream(
            options.stream_); // TODO: Change copy and assign structure!
        Ciphertext<Scheme::CKKS> result;
        result = cipher1;
        cipher1.switch_stream(
            old_stream); // TODO: Change copy and assign structure!

        int current_decomp_count = context_->Q_size - cipher1.depth_;
        cipher_mult_by_i_kernel<<<dim3((context_->n >> 8), current_decomp_count,
                                       2),
                                  256, 0, options.stream_>>>(
            result.data(), result.data(), context_->ntt_table_->data(),
            context_->modulus_->data(), context_->n_power);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        add(result, cipher0, result, options);

        Ciphertext<Scheme::CKKS> c1 =
            multiply_matrix(result, V_matrixs_rotated_encoded_,
                            diags_matrices_bsgs_, diags_matrices_bsgs_rot_n1_,
                            diags_matrices_bsgs_rot_n2_, galois_key, options);
        c1.encoding_ = encoding::COEFFICIENT;
        return c1;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::solo_slot_to_coeff(
        Ciphertext<Scheme::CKKS>& cipher, Galoiskey<Scheme::CKKS>& galois_key,
        const ExecutionOptions& options)
    {
        if (cipher.encoding_ != encoding::SLOT)
        {
            throw std::invalid_argument(
                "solo_slot_to_coeff expects slot-encoded ciphertext.");
        }

        Ciphertext<Scheme::CKKS> result =
            multiply_matrix(cipher, V_matrixs_rotated_encoded_,
                            diags_matrices_bsgs_, diags_matrices_bsgs_rot_n1_,
                            diags_matrices_bsgs_rot_n2_, galois_key, options);
        result.encoding_ = encoding::COEFFICIENT;

        return result;
    }

    /**
     * @brief Raises modulus from Q0 to full Q with optional key switching.
     */
    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::mod_up_from_q0(
        Ciphertext<Scheme::CKKS>& cipher,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        Ciphertext<Scheme::CKKS> cipher_after_ks;
        if (swk_dense_to_sparse != nullptr)
        {
            cipher_after_ks =
                operator_ciphertext(cipher.scale(), options.stream_);
            switchkey_ckks_method_II(cipher, cipher_after_ks,
                                     *swk_dense_to_sparse, options.stream_);
        }
        else
        {
            cipher_after_ks = cipher;
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = options.stream_};

        DeviceVector<Data64> cipher_intt_poly(2 * context_->n, options.stream_);
        input_storage_manager(
            cipher_after_ks,
            [&](Ciphertext<Scheme::CKKS>& cipher_temp)
            {
                gpuntt::GPU_INTT(cipher_after_ks.data(),
                                 cipher_intt_poly.data(),
                                 context_->intt_table_->data(),
                                 context_->modulus_->data(), cfg_intt, 2, 1);
            },
            options, false);

        Ciphertext<Scheme::CKKS> c_raised =
            operator_ciphertext(cipher.scale(), options.stream_);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        mod_raise_kernel<<<dim3((context_->n >> 8), context_->Q_size, 2), 256,
                           0, options.stream_>>>(
            cipher_intt_poly.data(), c_raised.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(c_raised.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                2 * context_->Q_size, context_->Q_size);

        if (swk_sparse_to_dense != nullptr)
        {
            switchkey_ckks_method_II(c_raised, c_raised, *swk_sparse_to_dense,
                                     options.stream_);
        }

        c_raised.encoding_ = cipher.encoding_;

        return c_raised;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::eval_mod(Ciphertext<Scheme::CKKS>& cipher,
                                       Relinkey<Scheme::CKKS>& relin_key,
                                       const ExecutionOptions& options)
    {
        return eval_mod(cipher, sine_poly_,
                        -0.5 / (std::pow(2.0, eval_mod_config_.double_angle_) *
                                (sine_poly_.b_ - sine_poly_.a_)),
                        relin_key, options);
    }

    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::eval_mod(
        Ciphertext<Scheme::CKKS>& cipher, const Polynomial& poly, double shift,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        double prev_scale = cipher.scale_;
        Ciphertext<Scheme::CKKS> cipher_taylor = cipher;
        cipher_taylor.scale_ = eval_mod_config_.scaling_factor_;

        double target_scale = eval_mod_config_.scaling_factor_;
        for (int i = 0; i < eval_mod_config_.double_angle_; i++)
        {
            int modulus_index = eval_mod_config_.level_start_ - poly.depth() -
                                eval_mod_config_.double_angle_ + i + 1;
            Data64 qi = context_->prime_vector_[modulus_index].value;
            target_scale = std::sqrt(target_scale * static_cast<double>(qi));
        }

        add_plain_inplace(cipher_taylor, shift, options);

        cipher_taylor = evaluate_poly(cipher_taylor, target_scale, scale_boot_,
                                      poly, relin_key, options);

        double sqrt2pi = eval_mod_config_.sqrt2pi_;
        for (int i = 0; i < eval_mod_config_.double_angle_; i++)
        {
            sqrt2pi *= sqrt2pi;
            multiply_inplace(cipher_taylor, cipher_taylor, options);
            relinearize_inplace(cipher_taylor, relin_key, options);
            add_inplace(cipher_taylor, cipher_taylor, options);
            add_plain_complex(cipher_taylor, Complex64(-sqrt2pi, 0.0),
                              cipher_taylor, options);
            rescale_inplace(cipher_taylor, options);
        }

        cipher_taylor.scale_ = prev_scale;
        return cipher_taylor;
    }

    __host__ void HEOperator<Scheme::CKKS>::gen_power(
        std::unordered_map<int, Ciphertext<Scheme::CKKS>>& cipher, int power,
        PolyType type, Relinkey<Scheme::CKKS>& relin_key,
        const ExecutionOptions& options)
    {
        if (cipher.count(power))
        {
            return;
        }

        bool is_pow2 = (power & (power - 1)) == 0;
        int a, b, c = 0;

        if (is_pow2)
        {
            a = power / 2;
            b = power / 2;
        }
        else
        {
            int k = int(std::ceil(std::log2(power))) - 1;
            a = (1 << k) - 1;
            b = power + 1 - (1 << k);

            if (type == PolyType::CHEBYSHEV)
            {
                c = std::abs(a - b);
            }
        }

        gen_power(cipher, a, type, relin_key, options);
        gen_power(cipher, b, type, relin_key, options);

        int x = cipher[a].level();
        int y = cipher[b].level();
        if (x != y)
        {
            Ciphertext<Scheme::CKKS> tmp = cipher[a];
            Ciphertext<Scheme::CKKS> tmp1 = cipher[b];

            if (x < y)
            {
                for (int i = x; i < y; i++)
                {
                    mod_drop_inplace(tmp1, options);
                }
            }
            else
            {
                for (int i = y; i < x; i++)
                {
                    mod_drop_inplace(tmp, options);
                }
            }
            multiply(tmp, tmp1, cipher[power], options);
        }
        else
        {
            multiply(cipher[a], cipher[b], cipher[power], options);
        }

        relinearize_inplace(cipher[power], relin_key, options);
        rescale_inplace(cipher[power], options);

        if (type == PolyType::CHEBYSHEV)
        {
            add_inplace(cipher[power], cipher[power], options);
            if (c == 0)
            {
                add_plain_complex(cipher[power], Complex64(-1.0, 0.0),
                                  cipher[power], options);
            }
            else
            {
                gen_power(cipher, c, type, relin_key, options);

                // T_c has a level more than the product. Multiplying it by 1
                // at the scale that a rescale turns into the scale of the
                // product makes both scales equal, which mod_drop does not.
                Ciphertext<Scheme::CKKS> tc = cipher[c];
                while (tc.level() > cipher[power].level() + 1)
                {
                    mod_drop_inplace(tc, options);
                }
                if (tc.level() == cipher[power].level() + 1)
                {
                    double qi = static_cast<double>(
                        context_->prime_vector_[tc.level()].value);
                    Ciphertext<Scheme::CKKS> scaled(context_, options);
                    multiply_plain(tc, 1.0, scaled,
                                   cipher[power].scale_ * qi / tc.scale_,
                                   options);
                    rescale_inplace(scaled, options);
                    tc = std::move(scaled);
                }
                while (cipher[power].level() > tc.level())
                {
                    mod_drop_inplace(cipher[power], options);
                }
                sub_inplace(cipher[power], tc, options);
            }
        }

        return;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_poly_from_polynomial_basis(
        double target_scale, int target_level, const Polynomial& pol,
        std::unordered_map<int, Ciphertext<Scheme::CKKS>>& powered_ciphers,
        const ExecutionOptions& options)
    {
        Ciphertext<Scheme::CKKS> result =
            operator_ciphertext(0, options.stream_);

        int current_decomp_count = target_level + 1;
        set_zero_cipher_ckks_poly<<<dim3((context_->n >> 8),
                                         current_decomp_count, 2),
                                    256, 0, options.stream_>>>(
            result.data(), context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = context_->scheme_;
        result.context_id_ = context_->context_id_;
        result.ring_size_ = context_->n;
        result.coeff_modulus_count_ = context_->Q_size;
        result.cipher_size_ = 2;
        result.depth_ = context_->Q_size - (target_level + 1);
        ;
        result.scale_ = target_scale;
        result.in_ntt_domain_ = true;
        result.encoding_ = encoding::SLOT;
        result.rescale_required_ = true;
        result.relinearization_required_ = false;
        result.ciphertext_generated_ = true;

        add_plain_complex(result, pol.coeffs_[0], result, options);

        for (int i = 1; i <= pol.degree(); i++)
        {
            Ciphertext<Scheme::CKKS> xi_term = powered_ciphers[i];

            DeviceVector<Data64> encoded_coeff_i(context_->Q_size
                                                 << context_->n_power);
            quick_ckks_encoder_constant_complex(pol.coeffs_[i],
                                                encoded_coeff_i.data(),
                                                target_scale / xi_term.scale_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            int current_decomp_count = context_->Q_size - xi_term.depth_;
            cipherplain_multiplication_kernel<<<dim3((context_->n >> 8),
                                                     current_decomp_count, 2),
                                                256, 0, options.stream_>>>(
                xi_term.data(), encoded_coeff_i.data(), xi_term.data(),
                context_->modulus_->data(), context_->n_power);
            xi_term.scale_ = target_scale;
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            int a = result.level();
            int b = xi_term.level();

            if (a != b)
            {
                Ciphertext<Scheme::CKKS> tmp = xi_term;
                Ciphertext<Scheme::CKKS> tmp1 = result;

                if (a < b)
                {
                    for (int j = a; j < b; j++)
                    {
                        mod_drop_inplace(tmp, options);
                    }
                }
                else
                {
                    for (int j = b; j < a; j++)
                    {
                        mod_drop_inplace(tmp1, options);
                    }
                }
                add(tmp, tmp1, result, options);
            }
            else
            {
                add_inplace(result, xi_term, options);
            }
        }

        return result;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_poly_recurse(
        int target_level, double target_scale, double base_scale,
        const Polynomial& pol, int log_split,
        std::unordered_map<int, Ciphertext<Scheme::CKKS>>& powered_ciphers,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        int degree = pol.degree();
        int split = 1 << log_split;

        if (degree < split)
        {
            if (pol.lead_ && log_split > 1 &&
                (pol.max_deg_ % (1 << (log_split + 1)) >
                 (1 << (log_split - 1))))
            {
                int log_degree = std::ceil(std::log2(degree));
                return evaluate_poly_recurse(
                    target_level, target_scale, base_scale, pol,
                    log_degree >> 1, powered_ciphers, relin_key, options);
            }

            if (pol.lead_)
            {
                target_scale = target_scale *
                               static_cast<double>(
                                   context_->prime_vector_[target_level].value);
            }

            return evaluate_poly_from_polynomial_basis(
                target_scale, target_level, pol, powered_ciphers, options);
        }

        int next_power = split;
        while (next_power < (degree >> 1) + 1)
        {
            next_power <<= 1;
        }

        auto [coeff_sq, coeff_sr] = pol.split_coeffs(next_power);

        Data64 current_qi;
        if (!pol.lead_)
        {
            current_qi = context_->prime_vector_[target_level + 1].value;
        }
        else
        {
            current_qi = context_->prime_vector_[target_level].value;
        }

        double next_target_scale = target_scale *
                                   static_cast<double>(current_qi) /
                                   powered_ciphers[next_power].scale_;

        Ciphertext<Scheme::CKKS> result_q = evaluate_poly_recurse(
            target_level + 1, next_target_scale, base_scale, coeff_sq,
            log_split, powered_ciphers, relin_key, options);

        if (result_q.scale_ >= base_scale / 2)
        {
            rescale_inplace(result_q, options);
        }

        int q_level = result_q.level();
        int power_level = powered_ciphers[next_power].level();
        if (q_level != power_level)
        {
            Ciphertext<Scheme::CKKS> tmp = powered_ciphers[next_power];
            Ciphertext<Scheme::CKKS> tmp1 = result_q;
            if (q_level < power_level)
            {
                for (int j = q_level; j < power_level; j++)
                {
                    mod_drop_inplace(tmp, options);
                }
            }
            else
            {
                for (int j = power_level; j < q_level; j++)
                {
                    mod_drop_inplace(tmp1, options);
                }
            }
            multiply(tmp1, tmp, result_q, options);
        }
        else
        {
            multiply_inplace(result_q, powered_ciphers[next_power], options);
        }

        relinearize_inplace(result_q, relin_key, options);
        // rescale_inplace(result_q, options);

        Ciphertext<Scheme::CKKS> result_r = evaluate_poly_recurse(
            result_q.level(), result_q.scale_, base_scale, coeff_sr, log_split,
            powered_ciphers, relin_key, options);

        int a = result_q.level();
        int b = result_r.level();
        if (a != b)
        {
            Ciphertext<Scheme::CKKS> tmp = result_q;
            Ciphertext<Scheme::CKKS> tmp1 = result_r;

            if (a < b)
            {
                for (int j = a; j < b; j++)
                {
                    mod_drop_inplace(tmp1, options);
                }
            }
            else
            {
                for (int j = b; j < a; j++)
                {
                    mod_drop_inplace(tmp, options);
                }
            }
            add(tmp, tmp1, result_q, options);
        }
        else
        {
            add_inplace(result_q, result_r, options);
        }

        return result_q;
    }

    static int optimal_split(int logDegree)
    {
        int logSplit = logDegree >> 1;
        int a = (1 << logSplit) + (1 << (logDegree - logSplit)) + logDegree -
                logSplit - 3;
        int b = (1 << (logSplit + 1)) + (1 << (logDegree - logSplit - 1)) +
                logDegree - logSplit - 4;
        if (a > b)
        {
            logSplit++;
        }
        return logSplit;
    }

    __host__ Ciphertext<Scheme::CKKS> HEOperator<Scheme::CKKS>::evaluate_poly(
        Ciphertext<Scheme::CKKS>& cipher, double target_scale,
        double base_scale, const Polynomial& pol,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher.stream();
        cipher.switch_stream(options.stream_);

        std::unordered_map<int, Ciphertext<Scheme::CKKS>> powered_ciphers;
        powered_ciphers[1] = cipher;

        // BSGS optimization: calculate optimal split point
        int poly_degree = pol.degree();
        int log_degree = std::ceil(std::log2(poly_degree + 1));
        int log_split = optimal_split(log_degree);

        // Baby-step: Generate powers x^1, x^2, ..., x^(2^logSplit - 1)
        for (int power = (1 << log_split) - 1; power >= 1; power--)
        {
            gen_power(powered_ciphers, power, pol.type_, relin_key, options);
        }

        // Giant-step: Generate powers x^(2^logSplit), x^(2^(logSplit+1)), ...,
        // x^(2^(logDegree-1))
        for (int i = log_split; i < log_degree; i++)
        {
            gen_power(powered_ciphers, 1 << i, pol.type_, relin_key, options);
        }

        int initial_target_level = cipher.level() - log_degree + 1;

        Ciphertext<Scheme::CKKS> result = evaluate_poly_recurse(
            initial_target_level, target_scale, base_scale, pol, log_split,
            powered_ciphers, relin_key, options);

        Data64 qi = context_->prime_vector_[result.level()].value;
        if (result.scale_ / static_cast<double>(qi) >= target_scale / 2.0)
        {
            rescale_inplace(result, options);
        }

        return result;
    }

    __host__ int HEOperator<Scheme::CKKS>::polynomial_depth(int degree,
                                                            bool chebyshev,
                                                            double a, double b)
    {
        if (degree < 1)
        {
            throw ParameterError(error_message(
                "Polynomial degree must be at least 1, got ", degree, "."));
        }

        int depth = 0;
        while ((1 << depth) <= degree)
        {
            depth++;
        }
        if (chebyshev && !(a == -1.0 && b == 1.0))
        {
            depth++;
        }
        return depth;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_in_basis(
        Ciphertext<Scheme::CKKS>& input1,
        const std::vector<double>& coefficients, PolyType type,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        std::vector<Complex64> coeffs(coefficients.size());
        for (size_t i = 0; i < coefficients.size(); i++)
        {
            coeffs[i] = Complex64(coefficients[i], 0.0);
        }
        int degree = static_cast<int>(coefficients.size()) - 1;
        Polynomial pol(degree, coeffs, true, type, -1.0, 1.0);

        // evaluate_poly moves its input to the stream of options.
        Ciphertext<Scheme::CKKS> input = input1;
        return evaluate_poly(input, input1.scale_, input1.scale_, pol,
                             relin_key, options);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_polynomial(
        Ciphertext<Scheme::CKKS>& input1,
        const std::vector<double>& coefficients,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        int degree = static_cast<int>(coefficients.size()) - 1;
        int depth = polynomial_depth(degree);
        if (input1.level() < depth)
        {
            throw LevelMismatchError(error_message(
                "A polynomial of degree ", degree, " needs ", depth,
                " levels, but the ciphertext has ", input1.level(), "."));
        }

        return evaluate_in_basis(input1, coefficients, PolyType::MONOMIAL,
                                 relin_key, options);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_chebyshev(
        Ciphertext<Scheme::CKKS>& input1,
        const std::vector<double>& coefficients, double a, double b,
        Relinkey<Scheme::CKKS>& relin_key, const ExecutionOptions& options)
    {
        input1.validate_context(context_);
        if (!(a < b))
        {
            throw ParameterError(
                error_message("The interval [", a, ", ", b, "] is empty."));
        }
        int degree = static_cast<int>(coefficients.size()) - 1;
        int depth = polynomial_depth(degree, true, a, b);
        if (input1.level() < depth)
        {
            throw LevelMismatchError(error_message(
                "A Chebyshev series of degree ", degree, " on [", a, ", ", b,
                "] needs ", depth, " levels, but the ciphertext has ",
                input1.level(), "."));
        }

        if (a == -1.0 && b == 1.0)
        {
            return evaluate_in_basis(input1, coefficients, PolyType::CHEBYSHEV,
                                     relin_key, options);
        }

        // Maps [a, b] to [-1, 1]. The constant is encoded with the scale of
        // the prime that rescale divides by, so the scale stays the same.
        Ciphertext<Scheme::CKKS> mapped(context_, options);
        double prime =
            static_cast<double>(context_->prime_vector_[input1.level()].value);
        multiply_plain(input1, 2.0 / (b - a), mapped, prime, options);
        rescale_inplace(mapped, options);
        mapped.scale_ = input1.scale_;
        add_plain_inplace(mapped, -(a + b) / (b - a), options);

        return evaluate_in_basis(mapped, coefficients, PolyType::CHEBYSHEV,
                                 relin_key, options);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEOperator<Scheme::CKKS>::evaluate_chebyshev(
        Ciphertext<Scheme::CKKS>& input1,
        const std::function<double(double)>& function, double a, double b,
        int degree, Relinkey<Scheme::CKKS>& relin_key,
        const ExecutionOptions& options)
    {
        if (degree < 1 || !(a < b))
        {
            throw ParameterError(error_message(
                "Chebyshev interpolation needs a degree of at least 1 and a "
                "non-empty interval, got degree ",
                degree, " on [", a, ", ", b, "]."));
        }
        return evaluate_chebyshev(
            input1, chebyshev_coefficients(function, a, b, degree), a, b,
            relin_key, options);
    }

    __host__ HEOperator<Scheme::CKKS>::Vandermonde::Vandermonde(
        const int poly_degree, const int CtoS_piece, const int StoC_piece,
        const double CtoS_Scaling, const double StoC_Scaling,
        const float CtoS_bsgs_ratio, const float StoC_bsgs_ratio)
    {
        poly_degree_ = poly_degree;
        num_slots_ = poly_degree_ >> 1;
        log_num_slots_ = int(log2l(num_slots_));

        CtoS_piece_ = CtoS_piece;
        StoC_piece_ = StoC_piece;

        CtoS_Scaling_ = CtoS_Scaling;
        StoC_Scaling_ = StoC_Scaling;

        generate_E_diagonals_index();
        generate_E_inv_diagonals_index();
        split_E();
        split_E_inv();

        generate_E_diagonals();
        generate_E_inv_diagonals();

        generate_V_n_lists(CtoS_bsgs_ratio, StoC_bsgs_ratio);

        generate_pre_comp_V();
        generate_pre_comp_V_inv();

        generate_key_indexs();
        key_indexs_ = unique_sort(key_indexs_);
    }

    __host__ void
    HEOperator<Scheme::CKKS>::Vandermonde::generate_E_diagonals_index()
    {
        bool first = true;
        for (int i = 1; i < (log_num_slots_ + 1); i++)
        {
            if (first)
            {
                int block_size = num_slots_ >> i;
                E_index_.push_back(0);
                E_index_.push_back(block_size);
                first = false;

                E_size_.push_back(2);
            }
            else
            {
                int block_size = num_slots_ >> i;
                E_index_.push_back(0);
                E_index_.push_back(block_size);
                E_index_.push_back(num_slots_ - block_size);

                E_size_.push_back(3);
            }
        }
    }

    __host__ void
    HEOperator<Scheme::CKKS>::Vandermonde::generate_E_inv_diagonals_index()
    {
        for (int i = log_num_slots_; 0 < i; i--)
        {
            if (i == 1)
            {
                int block_size = num_slots_ >> i;
                E_inv_index_.push_back(0);
                E_inv_index_.push_back(block_size);

                E_inv_size_.push_back(2);
            }
            else
            {
                int block_size = num_slots_ >> i;
                E_inv_index_.push_back(0);
                E_inv_index_.push_back(block_size);
                E_inv_index_.push_back(num_slots_ - block_size);

                E_inv_size_.push_back(3);
            }
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::split_E()
    {
        // E_splitted
        int k = log_num_slots_ / StoC_piece_;
        int m = log_num_slots_ % StoC_piece_;

        for (int i = 0; i < StoC_piece_; i++)
        {
            E_splitted_.push_back(k);
        }

        for (int i = 0; i < m; i++)
        {
            E_splitted_[i]++;
        }

        int counter = 0;
        for (int i = 0; i < StoC_piece_; i++)
        {
            std::vector<int> temp;
            for (int j = 0; j < E_splitted_[i]; j++)
            {
                int size = (counter == 0) ? 2 : 3;
                for (int k = 0; k < size; k++)
                {
                    temp.push_back(E_index_[counter]);
                    counter++;
                }
            }
            E_splitted_index_.push_back(temp);
        }

        int num_slots_mask = num_slots_ - 1;
        counter = 0;
        for (int k = 0; k < StoC_piece_; k++)
        {
            int matrix_count = E_splitted_[k];
            int L_m_loc = (k == 0) ? 2 : 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;
            std::vector<int> diag_index_temp;
            std::vector<int> iteration_temp;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    iteration_temp.push_back(E_size_[counter]);
                    for (int i = 0; i < E_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_splitted_index_[k][i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    iteration_temp.push_back(index_mul_sorted.size());
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            V_matrixs_index_.push_back(index_mul_sorted);
            E_splitted_diag_index_gpu_.push_back(diag_index_temp);
            E_splitted_iteration_gpu_.push_back(iteration_temp);
            counter += matrix_count;
        }

        std::vector<std::unordered_map<int, int>> dict_output_index;
        for (int k = 0; k < StoC_piece_; k++)
        {
            std::unordered_map<int, int> temp;
            for (int i = 0; i < V_matrixs_index_[k].size(); i++)
            {
                temp[V_matrixs_index_[k][i]] = i;
            }
            dict_output_index.push_back(temp);
        }

        counter = 0;
        for (int k = 0; k < StoC_piece_; k++)
        {
            int matrix_count = E_splitted_[k];
            int L_m_loc = (k == 0) ? 2 : 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;

            std::vector<int> temp_in_index;
            std::vector<int> temp_out_index;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    for (int i = 0; i < E_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_splitted_index_[k][i];
                        for (int j = 0; j < E_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        temp_in_index.push_back(
                            dict_output_index[k][R_m_İNDEX]);
                        for (int j = 0; j < E_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            counter += matrix_count;
            E_splitted_input_index_gpu_.push_back(temp_in_index);
            E_splitted_output_index_gpu_.push_back(temp_out_index);
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::split_E_inv()
    {
        // E_inv_splitted
        int k = log_num_slots_ / CtoS_piece_;
        int m = log_num_slots_ % CtoS_piece_;

        for (int i = 0; i < CtoS_piece_; i++)
        {
            E_inv_splitted_.push_back(k);
        }

        for (int i = 0; i < m; i++)
        {
            E_inv_splitted_[i]++;
        }

        int counter = 0;
        for (int i = 0; i < CtoS_piece_; i++)
        {
            std::vector<int> temp;
            for (int j = 0; j < E_inv_splitted_[i]; j++)
            {
                int size = (counter == (E_inv_index_.size() - 2)) ? 2 : 3;
                for (int k = 0; k < size; k++)
                {
                    temp.push_back(E_inv_index_[counter]);
                    counter++;
                }
            }
            E_inv_splitted_index_.push_back(temp);
        }

        int num_slots_mask = num_slots_ - 1;
        counter = 0;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            int matrix_count = E_inv_splitted_[k];

            int L_m_loc = 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;
            std::vector<int> diag_index_temp;
            std::vector<int> iteration_temp;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    iteration_temp.push_back(E_inv_size_[counter]);
                    for (int i = 0; i < E_inv_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_inv_splitted_index_[k][i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_inv_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    iteration_temp.push_back(index_mul_sorted.size());
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_inv_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            V_inv_matrixs_index_.push_back(index_mul_sorted);
            E_inv_splitted_diag_index_gpu_.push_back(diag_index_temp);
            E_inv_splitted_iteration_gpu_.push_back(iteration_temp);
            counter += matrix_count;
        }

        std::vector<std::unordered_map<int, int>> dict_output_index;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            std::unordered_map<int, int> temp;
            for (int i = 0; i < V_inv_matrixs_index_[k].size(); i++)
            {
                temp[V_inv_matrixs_index_[k][i]] = i;
            }
            dict_output_index.push_back(temp);
        }

        counter = 0;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            int matrix_count = E_inv_splitted_[k];
            int L_m_loc = 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;

            std::vector<int> temp_in_index;
            std::vector<int> temp_out_index;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    for (int i = 0; i < E_inv_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_inv_splitted_index_[k][i];
                        for (int j = 0; j < E_inv_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        temp_in_index.push_back(
                            dict_output_index[k][R_m_İNDEX]);
                        for (int j = 0; j < E_inv_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            counter += matrix_count;
            E_inv_splitted_input_index_gpu_.push_back(temp_in_index);
            E_inv_splitted_output_index_gpu_.push_back(temp_out_index);
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::generate_E_diagonals()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        heongpu::DeviceVector<Complex64> V_logn_diagnal(
            ((3 * log_num_slots_) - 1) << log_num_slots_);
        E_diagonal_generate_kernel<<<dim3(blokcount, log_num_slots_, 1),
                                     bloksize>>>(V_logn_diagnal.data(),
                                                 log_num_slots_);

        Complex64 scaling(std::pow(StoC_Scaling_, 1.0 / double(StoC_piece_)),
                          0.0);

        int matrix_counter = 0;
        for (int i = 0; i < StoC_piece_; i++)
        {
            heongpu::DeviceVector<int> diag_index_gpu(
                E_splitted_diag_index_gpu_[i]);
            heongpu::DeviceVector<int> input_index_gpu(
                E_splitted_input_index_gpu_[i]);
            heongpu::DeviceVector<int> output_index_gpu(
                E_splitted_output_index_gpu_[i]);

            heongpu::DeviceVector<Complex64> V_mul((V_matrixs_index_[i].size())
                                                   << log_num_slots_);
            cudaMemset(V_mul.data(), 0, V_mul.size() * sizeof(Complex64));

            int input_loc;
            if (i == 0)
            {
                input_loc = 0;
            }
            else
            {
                input_loc = ((3 * matrix_counter) - 1) << log_num_slots_;
            }

            int R_matrix_counter = 0;
            int output_index_counter = 0;

            for (int j = 0; j < (E_splitted_[i] - 1); j++)
            {
                heongpu::DeviceVector<Complex64> temp_result(
                    (V_matrixs_index_[i].size()) << log_num_slots_);
                cudaMemset(temp_result.data(), 0,
                           temp_result.size() * sizeof(Complex64));

                bool first_check1 = (i == 0) ? true : false;
                bool first_check2 = (j == 0) ? true : false;

                E_diagonal_matrix_mult_kernel<<<blokcount, bloksize>>>(
                    V_logn_diagnal.data() + input_loc, temp_result.data(),
                    V_mul.data(), diag_index_gpu.data(), input_index_gpu.data(),
                    output_index_gpu.data(), E_splitted_iteration_gpu_[i][j],
                    R_matrix_counter, output_index_counter, j, first_check1,
                    first_check2, log_num_slots_);

                V_mul = std::move(temp_result);

                R_matrix_counter += E_splitted_iteration_gpu_[i][j];
                output_index_counter += (E_splitted_iteration_gpu_[i][j] * 3);
            }

            if (StoC_Scaling_ != 1.0)
            {
                complex_vector_scale_kernel<<<
                    dim3(blokcount, V_matrixs_index_[i].size(), 1), bloksize>>>(
                    V_mul.data(), scaling, log_num_slots_);
            }

            V_matrixs_.push_back(std::move(V_mul));
            matrix_counter += E_splitted_[i];
        }
    }

    __host__ void
    HEOperator<Scheme::CKKS>::Vandermonde::generate_E_inv_diagonals()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        heongpu::DeviceVector<Complex64> V_inv_logn_diagnal(
            ((3 * log_num_slots_) - 1) << log_num_slots_);
        E_diagonal_inverse_generate_kernel<<<dim3(blokcount, log_num_slots_, 1),
                                             bloksize>>>(
            V_inv_logn_diagnal.data(), log_num_slots_);

        Complex64 scaling(std::pow(CtoS_Scaling_, 1.0 / double(CtoS_piece_)),
                          0.0);

        int matrix_counter = 0;
        for (int i = 0; i < CtoS_piece_; i++)
        {
            heongpu::DeviceVector<int> diag_index_gpu(
                E_inv_splitted_diag_index_gpu_[i]);
            heongpu::DeviceVector<int> input_index_gpu(
                E_inv_splitted_input_index_gpu_[i]);
            heongpu::DeviceVector<int> output_index_gpu(
                E_inv_splitted_output_index_gpu_[i]);

            heongpu::DeviceVector<Complex64> V_mul(
                (V_inv_matrixs_index_[i].size()) << log_num_slots_);
            cudaMemset(V_mul.data(), 0, V_mul.size() * sizeof(Complex64));

            int input_loc = (3 * matrix_counter) << log_num_slots_;
            int R_matrix_counter = 0;
            int output_index_counter = 0;

            for (int j = 0; j < (E_inv_splitted_[i] - 1); j++)
            {
                heongpu::DeviceVector<Complex64> temp_result(
                    (V_inv_matrixs_index_[i].size()) << log_num_slots_);
                cudaMemset(temp_result.data(), 0,
                           temp_result.size() * sizeof(Complex64));
                bool first_check = (j == 0) ? true : false;
                bool last_check = ((i == (CtoS_piece_ - 1)) &&
                                   (j == (E_inv_splitted_[i] - 2)))
                                      ? true
                                      : false;

                E_diagonal_inverse_matrix_mult_kernel<<<blokcount, bloksize>>>(
                    V_inv_logn_diagnal.data() + input_loc, temp_result.data(),
                    V_mul.data(), diag_index_gpu.data(), input_index_gpu.data(),
                    output_index_gpu.data(),
                    E_inv_splitted_iteration_gpu_[i][j], R_matrix_counter,
                    output_index_counter, j, first_check, last_check,
                    log_num_slots_);

                V_mul = std::move(temp_result);
                R_matrix_counter += E_inv_splitted_iteration_gpu_[i][j];
                output_index_counter +=
                    (E_inv_splitted_iteration_gpu_[i][j] * 3);
            }

            if (CtoS_Scaling_ != 1.0)
            {
                complex_vector_scale_kernel<<<
                    dim3(blokcount, V_inv_matrixs_index_[i].size(), 1),
                    bloksize>>>(V_mul.data(), scaling, log_num_slots_);
            }

            V_inv_matrixs_.push_back(std::move(V_mul));
            matrix_counter += E_inv_splitted_[i];
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::generate_V_n_lists(
        float CtoS_bsgs_ratio, float StoC_bsgs_ratio)
    {
        for (int i = 0; i < StoC_piece_; i++)
        {
            std::vector<int> rot_n1, rot_n2;
            std::vector<std::vector<int>> result =
                heongpu::seperate_func(V_matrixs_index_[i], num_slots_, rot_n1,
                                       rot_n2, StoC_bsgs_ratio);

            diags_matrices_bsgs_.push_back(std::move(result));
            diags_matrices_bsgs_rot_n1_.push_back(std::move(rot_n1));
            diags_matrices_bsgs_rot_n2_.push_back(std::move(rot_n2));
        }

        for (int i = 0; i < CtoS_piece_; i++)
        {
            std::vector<int> rot_n1, rot_n2;
            std::vector<std::vector<int>> result =
                heongpu::seperate_func(V_inv_matrixs_index_[i], num_slots_,
                                       rot_n1, rot_n2, CtoS_bsgs_ratio);

            diags_matrices_inv_bsgs_.push_back(std::move(result));
            diags_matrices_inv_bsgs_rot_n1_.push_back(std::move(rot_n1));
            diags_matrices_inv_bsgs_rot_n2_.push_back(std::move(rot_n2));
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::generate_pre_comp_V()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        for (int m = 0; m < StoC_piece_; m++)
        {
            heongpu::DeviceVector<Complex64> temp_rotated(
                (V_matrixs_index_[m].size()) << log_num_slots_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                int real_shift = -(diags_matrices_bsgs_rot_n1_[m][j]);
                for (int i = 0; i < diags_matrices_bsgs_[m][j].size(); i++)
                {
                    int location = (counter << log_num_slots_);

                    vector_rotate_kernel<<<blokcount, bloksize>>>(
                        V_matrixs_[m].data() + location,
                        temp_rotated.data() + location, real_shift,
                        log_num_slots_);

                    counter++;
                }
            }

            V_matrixs_rotated_.push_back(std::move(temp_rotated));
        }
    }

    __host__ void
    HEOperator<Scheme::CKKS>::Vandermonde::generate_pre_comp_V_inv()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        for (int m = 0; m < CtoS_piece_; m++)
        {
            heongpu::DeviceVector<Complex64> temp_rotated(
                (V_inv_matrixs_index_[m].size()) << log_num_slots_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_inv_bsgs_[m].size(); j++)
            {
                int real_shift = -(diags_matrices_inv_bsgs_rot_n1_[m][j]);
                for (int i = 0; i < diags_matrices_inv_bsgs_[m][j].size(); i++)
                {
                    int location = (counter << log_num_slots_);

                    vector_rotate_kernel<<<blokcount, bloksize>>>(
                        V_inv_matrixs_[m].data() + location,
                        temp_rotated.data() + location, real_shift,
                        log_num_slots_);

                    counter++;
                }
            }

            V_inv_matrixs_rotated_.push_back(std::move(temp_rotated));
        }
    }

    __host__ void HEOperator<Scheme::CKKS>::Vandermonde::generate_key_indexs()
    {
        for (int m = 0; m < CtoS_piece_; m++)
        {
            for (const auto& index : diags_matrices_inv_bsgs_rot_n1_[m])
            {
                if (index != 0 &&
                    std::find(key_indexs_.begin(), key_indexs_.end(), index) ==
                        key_indexs_.end())
                {
                    key_indexs_.push_back(index);
                }
            }

            for (int j = 0; j < diags_matrices_inv_bsgs_[m].size(); j++)
            {
                for (const int& index : diags_matrices_inv_bsgs_[m][j])
                {
                    int index_ = index - diags_matrices_inv_bsgs_rot_n1_[m][j];
                    if (index_ != 0 &&
                        std::find(key_indexs_.begin(), key_indexs_.end(),
                                  index_) == key_indexs_.end())
                    {
                        key_indexs_.push_back(index_);
                    }
                }
            }
        }

        for (int m = 0; m < StoC_piece_; m++)
        {
            for (const auto& index : diags_matrices_bsgs_rot_n1_[m])
            {
                if (index != 0 &&
                    std::find(key_indexs_.begin(), key_indexs_.end(), index) ==
                        key_indexs_.end())
                {
                    key_indexs_.push_back(index);
                }
            }

            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                for (const int& index : diags_matrices_bsgs_[m][j])
                {
                    int index_ = index - diags_matrices_bsgs_rot_n1_[m][j];
                    if (index_ != 0 &&
                        std::find(key_indexs_.begin(), key_indexs_.end(),
                                  index_) == key_indexs_.end())
                    {
                        key_indexs_.push_back(index_);
                    }
                }
            }
        }
    }

    __host__ HEOperator<Scheme::CKKS>::Polynomial::Polynomial(
        int max_deg, const std::vector<Complex64>& coeffs, bool lead,
        PolyType type, double a, double b)
        : max_deg_(max_deg), coeffs_(coeffs), lead_(lead), type_(type), a_(a),
          b_(b)
    {
    }

    __host__ HEOperator<Scheme::CKKS>::Polynomial
    HEOperator<Scheme::CKKS>::generate_eval_mod_poly(
        const EvalModConfig& config, int max_deg)
    {
        double a, b;
        std::vector<Complex64> coeffs;

        if (config.sine_type_ == SineType::COS1)
        {
            a = -static_cast<double>(config.K_) /
                std::pow(2.0, config.double_angle_);
            b = static_cast<double>(config.K_) /
                std::pow(2.0, config.double_angle_);

            // Use high-precision ApproximateCos from cosine_approx.cu
            auto high_prec_coeffs =
                ApproximateCos(config.K_, max_deg, config.message_ratio_,
                               config.double_angle_);
            coeffs.resize(high_prec_coeffs.size());
            for (size_t i = 0; i < high_prec_coeffs.size(); i++)
            {
                coeffs[i] = Complex64(high_prec_coeffs[i].real(),
                                      high_prec_coeffs[i].imag());
            }
        }
        else
        {
            throw std::invalid_argument("Unsupported sine type!");
        }

        return Polynomial(max_deg, coeffs, true, config.poly_type_, a, b);
    }

    __host__ int HEOperator<Scheme::CKKS>::Polynomial::degree() const
    {
        return coeffs_.size() - 1;
    }

    __host__ int HEOperator<Scheme::CKKS>::Polynomial::depth() const
    {
        return static_cast<int>(std::ceil(std::log2(coeffs_.size())));
    }

    __host__ std::pair<HEOperator<Scheme::CKKS>::Polynomial,
                       HEOperator<Scheme::CKKS>::Polynomial>
    HEOperator<Scheme::CKKS>::Polynomial::split_coeffs(int split) const
    {
        int max_deg_r = split - 1;
        if (max_deg_ != degree())
        {
            max_deg_r = max_deg_ - (degree() - split + 1);
        }

        // Create coeffs_r: coefficients [0, split)
        std::vector<Complex64> coeffs_r(coeffs_.begin(),
                                        coeffs_.begin() + split);
        Polynomial poly_r(max_deg_r, coeffs_r, false, type_, a_, b_);

        // Create coeffs_q: coefficients [split, degree()]
        int q_size = degree() - split + 1;
        std::vector<Complex64> coeffs_q(q_size, coeffs_[split]);

        if (type_ == PolyType::MONOMIAL)
        {
            // For monomial type: simply copy coeffs[split+1:] as-is
            for (int i = split + 1; i <= degree(); i++)
            {
                coeffs_q[i - split] = coeffs_[i];
            }
        }
        else if (type_ == PolyType::CHEBYSHEV)
        {
            // For Chebyshev type: apply the transformation
            int i = split + 1;
            int j = 1;
            while (i <= degree())
            {
                coeffs_q[i - split] = Complex64(2.0, 0.0) * coeffs_[i];
                poly_r.coeffs_[split - j] =
                    poly_r.coeffs_[split - j] - coeffs_[i];
                i++;
                j++;
            }
        }

        Polynomial poly_q(max_deg_, coeffs_q, lead_, type_, a_, b_);

        return std::make_pair(poly_q, poly_r);
    }

    HEArithmeticOperator<Scheme::CKKS>::HEArithmeticOperator(
        HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder)
        : HEOperator<Scheme::CKKS>(context, encoder)
    {
    }

    __host__ void
    HEArithmeticOperator<Scheme::CKKS>::generate_encoding_transform_context(
        CKKSEncodingTransformContext& transform_context,
        const CKKSEncodingTransformConfig& config)
    {
        if (config.CtoS_piece_ < 2 || config.CtoS_piece_ > 5)
        {
            throw std::out_of_range("CtoS_piece must be in range [2, 5]");
        }

        if (config.StoC_piece_ < 2 || config.StoC_piece_ > 5)
        {
            throw std::out_of_range("StoC_piece must be in range [2, 5]");
        }

        transform_context.CtoS_piece_ = config.CtoS_piece_;
        transform_context.StoC_piece_ = config.StoC_piece_;

        int ctos_level =
            (config.CtoS_start_level_ < 0) ? 0 : config.CtoS_start_level_;
        int stoc_level = (config.StoC_start_level_ < 0)
                             ? (ctos_level + transform_context.CtoS_piece_)
                             : config.StoC_start_level_;

        if (ctos_level < 0 || ctos_level >= context_->Q_size)
        {
            throw std::out_of_range(
                "CtoS_start_level must be in range [0, Q_size - 1]");
        }

        if (stoc_level < 0 || stoc_level >= context_->Q_size)
        {
            throw std::out_of_range(
                "StoC_start_level must be in range [0, Q_size - 1]");
        }

        if ((context_->Q_size - ctos_level) <= transform_context.CtoS_piece_)
        {
            throw std::out_of_range(
                "CtoS_start_level is too deep for selected CtoS_piece.");
        }

        if ((context_->Q_size - stoc_level) <= transform_context.StoC_piece_)
        {
            throw std::out_of_range(
                "StoC_start_level is too deep for selected StoC_piece.");
        }

        transform_context.CtoS_level_ = ctos_level;
        transform_context.StoC_level_ = stoc_level;

        EncodingMatrixConfig cts_config(LinearTransformType::COEFFS_TO_SLOTS,
                                        context_->Q_size - 1 - ctos_level);
        EncodingMatrixConfig stc_config(LinearTransformType::SLOTS_TO_COEFFS,
                                        context_->Q_size - 1 - stoc_level);

        // CtoS output is (c + conj(c)), so its 1/2 is folded into the matrix.
        Vandermonde matrix_gen(context_->n, transform_context.CtoS_piece_,
                               transform_context.StoC_piece_, 0.5, 1.0,
                               cts_config.bsgs_ratio_, stc_config.bsgs_ratio_);

        transform_context.V_matrixs_rotated_encoded_ =
            encode_V_matrixs(matrix_gen, stc_config.level_start_);
        transform_context.V_inv_matrixs_rotated_encoded_ =
            encode_V_inv_matrixs(matrix_gen, cts_config.level_start_);

        transform_context.diags_matrices_bsgs_ =
            matrix_gen.diags_matrices_bsgs_;
        transform_context.diags_matrices_inv_bsgs_ =
            matrix_gen.diags_matrices_inv_bsgs_;

        transform_context.diags_matrices_bsgs_rot_n1_ =
            matrix_gen.diags_matrices_bsgs_rot_n1_;
        transform_context.diags_matrices_inv_bsgs_rot_n1_ =
            matrix_gen.diags_matrices_inv_bsgs_rot_n1_;

        transform_context.diags_matrices_bsgs_rot_n2_ =
            matrix_gen.diags_matrices_bsgs_rot_n2_;
        transform_context.diags_matrices_inv_bsgs_rot_n2_ =
            matrix_gen.diags_matrices_inv_bsgs_rot_n2_;

        transform_context.key_indexs_ = matrix_gen.key_indexs_;

        transform_context.generated_ = true;
    }

    __host__ void
    HEArithmeticOperator<Scheme::CKKS>::generate_bootstrapping_params(
        const double scale, const BootstrappingConfig& config,
        const arithmetic_bootstrapping_type& boot_type)
    {
        if (!boot_context_generated_)
        {
            scale_boot_ = scale;
            CtoS_piece_ = config.CtoS_piece_;
            StoC_piece_ = config.StoC_piece_;

            // Copy configuration objects
            cts_config_ = config.cts_config_;
            stc_config_ = config.stc_config_;
            eval_mod_config_ = config.eval_mod_config_;

            // Set Q if not already set in config
            if (eval_mod_config_.Q_ == 0)
            {
                eval_mod_config_.Q_ = context_->prime_vector_[0].value;
            }
            // Set scaling_factor based on Q0's bit length
            if (eval_mod_config_.scaling_factor_ == 0.0)
            {
                int bit_length = calculate_bit_count(
                    context_->prime_vector_[eval_mod_config_.level_start_]
                        .value);
                eval_mod_config_.scaling_factor_ =
                    static_cast<double>(1ULL << bit_length);
            }

            eval_mod_config_.q_diff_ =
                static_cast<double>(eval_mod_config_.Q_) /
                std::pow(2.0, std::round(std::log2(
                                  static_cast<double>(eval_mod_config_.Q_))));
            // double CtoS_Scaling = 1.0
            // /(double(eval_mod_config_.K_)*double(context_->n)*eval_mod_config_.q_diff_);
            double CtoS_Scaling =
                0.5 / (double(eval_mod_config_.K_) * eval_mod_config_.q_diff_);
            double StoC_Scaling =
                scale_boot_ / (eval_mod_config_.scaling_factor_ /
                               eval_mod_config_.message_ratio_);

            // Slim runs SlotToCoeff first, so the output scaling is folded
            // into the EvalMod amplitude instead of the StoC matrix.
            double eval_mod_amplitude = eval_mod_config_.q_diff_ / (2.0 * M_PI);
            switch (static_cast<int>(boot_type))
            {
                case 1: // REGULAR_BOOTSTRAPPING
                    break;
                case 2: // SLIM_BOOTSTRAPPING
                    eval_mod_amplitude *= StoC_Scaling;
                    StoC_Scaling = 1.0;
                    break;
                default:
                    throw std::invalid_argument("Invalid Bootstrapping Type");
                    break;
            }

            eval_mod_config_.sqrt2pi_ =
                std::pow(eval_mod_amplitude,
                         1.0 / std::pow(2.0, eval_mod_config_.double_angle_));

            Vandermonde matrix_gen(
                context_->n, CtoS_piece_, StoC_piece_, CtoS_Scaling,
                StoC_Scaling, cts_config_.bsgs_ratio_, stc_config_.bsgs_ratio_);

            V_matrixs_rotated_encoded_ =
                encode_V_matrixs(matrix_gen, stc_config_.level_start_);
            V_inv_matrixs_rotated_encoded_ =
                encode_V_inv_matrixs(matrix_gen, cts_config_.level_start_);

            V_matrixs_index_ = matrix_gen.V_matrixs_index_;
            V_inv_matrixs_index_ = matrix_gen.V_inv_matrixs_index_;

            diags_matrices_bsgs_ = matrix_gen.diags_matrices_bsgs_;
            diags_matrices_inv_bsgs_ = matrix_gen.diags_matrices_inv_bsgs_;

            diags_matrices_bsgs_rot_n1_ =
                matrix_gen.diags_matrices_bsgs_rot_n1_;
            diags_matrices_inv_bsgs_rot_n1_ =
                matrix_gen.diags_matrices_inv_bsgs_rot_n1_;

            diags_matrices_bsgs_rot_n2_ =
                matrix_gen.diags_matrices_bsgs_rot_n2_;
            diags_matrices_inv_bsgs_rot_n2_ =
                matrix_gen.diags_matrices_inv_bsgs_rot_n2_;

            key_indexs_ = matrix_gen.key_indexs_;

            sine_poly_ = generate_eval_mod_poly(eval_mod_config_,
                                                eval_mod_config_.sine_deg_);

            if (eval_mod_config_.sine_type_ == SineType::COS1)
            {
                for (int i = 0; i < sine_poly_.coeffs_.size(); i++)
                {
                    sine_poly_.coeffs_[i] =
                        sine_poly_.coeffs_[i] *
                        Complex64(eval_mod_config_.sqrt2pi_, 0.0);
                }
            }

            boot_context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Bootstrapping parameters is locked after "
                                     "generation and cannot be modified.");
        }

        cudaDeviceSynchronize();
    }

    /**
     * @brief Regular bootstrapping procedure based on the algorithm from
     *        "Efficient Bootstrapping for Approximate Homomorphic Encryption
     * with Non-Sparse Keys" (https://eprint.iacr.org/2020/1203.pdf)
     * @param input1 Input ciphertext at maximum depth (only 1 modulus
     * remaining)
     * @param galois_key Galois keys for rotation operations in
     * CoeffToSlot/SlotToCoeff
     * @param relin_key Relinearization key for reducing ciphertext size during
     * EvalMod
     * @param swk_dense_to_sparse Optional switch key from dense to sparse
     * representation
     * @param swk_sparse_to_dense Optional switch key from sparse back to dense
     * representation
     * @param options Execution options (stream, storage type, etc.)
     * @return Bootstrapped ciphertext with refreshed noise and full modulus
     * chain
     */
    __host__ void
    HEArithmeticOperator<Scheme::CKKS>::generate_bootstrapping_params(
        const CKKSBootstrappingPreset& preset)
    {
        if (preset.arithmetic_type == arithmetic_bootstrapping_type::NONE)
        {
            throw ParameterError(
                "The preset is not a regular or slim bootstrapping preset!");
        }
        generate_bootstrapping_params(preset.scale, preset.config,
                                      preset.arithmetic_type);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEArithmeticOperator<Scheme::CKKS>::regular_bootstrapping(
        Ciphertext<Scheme::CKKS>& input1, Galoiskey<Scheme::CKKS>& galois_key,
        Relinkey<Scheme::CKKS>& relin_key,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        input1.validate_context(context_);

        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // An input above level 0 is brought down to it.
        if (input1.level() > 0)
        {
            Ciphertext<Scheme::CKKS> dropped = input1;
            while (dropped.level() > 0)
            {
                mod_drop_inplace(dropped, options);
            }
            return regular_bootstrapping(dropped, galois_key, relin_key,
                                         swk_dense_to_sparse,
                                         swk_sparse_to_dense, options);
        }

        // Raise modulus
        int current_decomp_count = context_->Q_size - input1.depth_;
        if (current_decomp_count != 1)
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled should be at max! The ciphertext has ",
                current_decomp_count, " moduli, expected ", 1, "."));
        }

        Ciphertext<Scheme::CKKS> scaled_input1 = input1;
        double q0OverMessageRatio = std::exp2(std::round(
            std::log2(static_cast<double>(context_->prime_vector_[0].value) /
                      eval_mod_config_.message_ratio_)));

        scale_up(input1, std::round(q0OverMessageRatio / input1.scale()),
                 scaled_input1, options);

        if (std::round((static_cast<double>(context_->prime_vector_[0].value) /
                        eval_mod_config_.message_ratio_) /
                       scaled_input1.scale()) > 1.0)
        {
            scale_up(scaled_input1,
                     std::round((static_cast<double>(
                                     context_->prime_vector_[0].value) /
                                 eval_mod_config_.message_ratio_) /
                                scaled_input1.scale()),
                     scaled_input1, options);
        }

        Ciphertext<Scheme::CKKS> c_raised = mod_up_from_q0(
            scaled_input1, swk_dense_to_sparse, swk_sparse_to_dense, options);

        if ((eval_mod_config_.scaling_factor_ /
             eval_mod_config_.message_ratio_) /
                c_raised.scale() >
            1.0)
        {
            scale_up(c_raised,
                     std::round((eval_mod_config_.scaling_factor_ /
                                 eval_mod_config_.message_ratio_) /
                                c_raised.scale()),
                     c_raised, options);
        }
        c_raised.encoding_ = encoding::COEFFICIENT;

        // Coeff to slot
        std::vector<heongpu::Ciphertext<Scheme::CKKS>> enc_results =
            coeff_to_slot(c_raised, galois_key, options); // c_raised

        Ciphertext<Scheme::CKKS> ciph_sin0 =
            eval_mod(enc_results[0], relin_key, options);
        Ciphertext<Scheme::CKKS> ciph_sin1 =
            eval_mod(enc_results[1], relin_key, options);
        ciph_sin0.scale_ = scale_boot_;
        ciph_sin1.scale_ = scale_boot_;

        // Slot to coeff
        Ciphertext<Scheme::CKKS> StoC_results =
            slot_to_coeff(ciph_sin0, ciph_sin1, galois_key, options);
        StoC_results.scale_ = scale_boot_;
        // Bootstrapping API returns a regular CKKS slot ciphertext.
        StoC_results.encoding_ = encoding::SLOT;

        return StoC_results;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEArithmeticOperator<Scheme::CKKS>::slim_bootstrapping(
        Ciphertext<Scheme::CKKS>& input1, Galoiskey<Scheme::CKKS>& galois_key,
        Relinkey<Scheme::CKKS>& relin_key,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        input1.validate_context(context_);

        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // An input above the SlotToCoeff start level is brought down to it.
        if (input1.level() > stc_config_.level_start_)
        {
            Ciphertext<Scheme::CKKS> dropped = input1;
            while (dropped.level() > stc_config_.level_start_)
            {
                mod_drop_inplace(dropped, options);
            }
            return slim_bootstrapping(dropped, galois_key, relin_key,
                                      swk_dense_to_sparse, swk_sparse_to_dense,
                                      options);
        }

        int current_decomp_count = context_->Q_size - input1.depth_;
        if (current_decomp_count != (stc_config_.level_start_ + 1))
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled should be at max! The ciphertext has ",
                current_decomp_count, " moduli, expected ",
                stc_config_.level_start_ + 1, "."));
        }

        // Slot to coeff
        Ciphertext<Scheme::CKKS> StoC_results =
            solo_slot_to_coeff(input1, galois_key, options);

        // Raise modulus
        Ciphertext<Scheme::CKKS> scaled_input1 = StoC_results;
        double q0OverMessageRatio = std::exp2(std::round(
            std::log2(static_cast<double>(context_->prime_vector_[0].value) /
                      eval_mod_config_.message_ratio_)));

        scale_up(StoC_results,
                 std::round(q0OverMessageRatio / StoC_results.scale()),
                 scaled_input1, options);

        if (std::round((static_cast<double>(context_->prime_vector_[0].value) /
                        eval_mod_config_.message_ratio_) /
                       scaled_input1.scale()) > 1.0)
        {
            scale_up(scaled_input1,
                     std::round((static_cast<double>(
                                     context_->prime_vector_[0].value) /
                                 eval_mod_config_.message_ratio_) /
                                scaled_input1.scale()),
                     scaled_input1, options);
        }

        Ciphertext<Scheme::CKKS> c_raised = mod_up_from_q0(
            scaled_input1, swk_dense_to_sparse, swk_sparse_to_dense, options);

        if ((eval_mod_config_.scaling_factor_ /
             eval_mod_config_.message_ratio_) /
                c_raised.scale() >
            1.0)
        {
            scale_up(c_raised,
                     std::round((eval_mod_config_.scaling_factor_ /
                                 eval_mod_config_.message_ratio_) /
                                c_raised.scale()),
                     c_raised, options);
        }
        c_raised.encoding_ = encoding::COEFFICIENT;

        // Coeff to slot
        Ciphertext<Scheme::CKKS> CtoS_results =
            solo_coeff_to_slot(c_raised, galois_key, options);

        Ciphertext<Scheme::CKKS> ciph_sin =
            eval_mod(CtoS_results, relin_key, options);
        ciph_sin.scale_ = scale_boot_;
        ciph_sin.encoding_ = encoding::SLOT;

        return ciph_sin;
    }

    HELogicOperator<Scheme::CKKS>::HELogicOperator(
        HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder,
        double scale)
        : HEOperator<Scheme::CKKS>(context, encoder)
    {
        if (scale == 0.0)
        {
            throw std::invalid_argument(
                "Scale can not be zero for CKKS Scheme");
        }

        double constant_1 = 1.0;
        encoded_constant_one_ =
            DeviceVector<Data64>(context_->Q_size << context_->n_power);
        quick_ckks_encoder_constant_double(constant_1,
                                           encoded_constant_one_.data(), scale);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HELogicOperator<Scheme::CKKS>::generate_bootstrapping_params(
        const CKKSBootstrappingPreset& preset)
    {
        if (preset.logic_type == logic_bootstrapping_type::NONE)
        {
            throw ParameterError(
                "The preset is not a bit or gate bootstrapping preset!");
        }
        generate_bootstrapping_params(preset.scale, preset.config,
                                      preset.logic_type);
    }

    __host__ void HELogicOperator<Scheme::CKKS>::generate_bootstrapping_params(
        const double scale, const BootstrappingConfig& config,
        const logic_bootstrapping_type& boot_type)
    {
        if (!boot_context_generated_)
        {
            scale_boot_ = scale;
            CtoS_piece_ = config.CtoS_piece_;
            StoC_piece_ = config.StoC_piece_;

            cts_config_ = config.cts_config_;
            stc_config_ = config.stc_config_;
            eval_mod_config_ = config.eval_mod_config_;

            // Inputs are placed at q0/period (bit: b*q0/2, gate: (b1+b2)*q0/3)
            // and each function is constant +
            // amplitude*cos(2*pi*(x-shift)/period).
            int period;
            double amplitude;
            std::vector<double> shifts, constants;
            switch (static_cast<int>(boot_type))
            {
                case 1: // BIT_BOOTSTRAPPING
                    period = 2;
                    amplitude = 0.5;
                    shifts = {1.0};
                    constants = {0.5};
                    break;
                case 2: // GATE_BOOTSTRAPPING (AND, OR, XOR, NAND, NOR, XNOR)
                    period = 3;
                    amplitude = 2.0 / 3.0;
                    shifts = {-1.0, 1.5, 1.0, 0.5, 0.0, -0.5};
                    constants = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0,
                                 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0};
                    break;
                default:
                    throw std::invalid_argument("Invalid Bootstrapping Type");
                    break;
            }
            double dev = eval_mod_config_.message_ratio_;
            eval_mod_config_.message_ratio_ = period;

            if (eval_mod_config_.Q_ == 0)
            {
                eval_mod_config_.Q_ = context_->prime_vector_[0].value;
            }
            if (eval_mod_config_.scaling_factor_ == 0.0)
            {
                int bit_length = calculate_bit_count(
                    context_->prime_vector_[eval_mod_config_.level_start_]
                        .value);
                eval_mod_config_.scaling_factor_ =
                    static_cast<double>(1ULL << bit_length);
            }

            eval_mod_config_.q_diff_ =
                static_cast<double>(eval_mod_config_.Q_) /
                std::pow(2.0, std::round(std::log2(
                                  static_cast<double>(eval_mod_config_.Q_))));
            eval_mod_config_.sqrt2pi_ = std::pow(
                amplitude * scale_boot_ / eval_mod_config_.scaling_factor_,
                1.0 / std::pow(2.0, eval_mod_config_.double_angle_));

            double CtoS_Scaling =
                0.5 / (double(eval_mod_config_.K_) * eval_mod_config_.q_diff_);

            Vandermonde matrix_gen(context_->n, CtoS_piece_, StoC_piece_,
                                   CtoS_Scaling, 1.0, cts_config_.bsgs_ratio_,
                                   stc_config_.bsgs_ratio_);

            V_matrixs_rotated_encoded_ =
                encode_V_matrixs(matrix_gen, stc_config_.level_start_);
            V_inv_matrixs_rotated_encoded_ =
                encode_V_inv_matrixs(matrix_gen, cts_config_.level_start_);

            V_matrixs_index_ = matrix_gen.V_matrixs_index_;
            V_inv_matrixs_index_ = matrix_gen.V_inv_matrixs_index_;

            diags_matrices_bsgs_ = matrix_gen.diags_matrices_bsgs_;
            diags_matrices_inv_bsgs_ = matrix_gen.diags_matrices_inv_bsgs_;

            diags_matrices_bsgs_rot_n1_ =
                matrix_gen.diags_matrices_bsgs_rot_n1_;
            diags_matrices_inv_bsgs_rot_n1_ =
                matrix_gen.diags_matrices_inv_bsgs_rot_n1_;

            diags_matrices_bsgs_rot_n2_ =
                matrix_gen.diags_matrices_bsgs_rot_n2_;
            diags_matrices_inv_bsgs_rot_n2_ =
                matrix_gen.diags_matrices_inv_bsgs_rot_n2_;

            key_indexs_ = matrix_gen.key_indexs_;

            int K = period * eval_mod_config_.K_;
            double b = static_cast<double>(eval_mod_config_.K_) /
                       std::pow(2.0, eval_mod_config_.double_angle_);
            for (size_t i = 0; i < shifts.size(); i++)
            {
                auto cos_coeffs = ApproximateCos(K, 2 * K - 2, dev,
                                                 eval_mod_config_.double_angle_,
                                                 period, shifts[i]);
                std::vector<Complex64> coeffs(cos_coeffs.size());
                for (size_t j = 0; j < cos_coeffs.size(); j++)
                {
                    coeffs[j] = Complex64(
                        cos_coeffs[j].real() * eval_mod_config_.sqrt2pi_, 0.0);
                }

                logic_polys_.push_back(Polynomial(2 * K - 2, coeffs, true,
                                                  eval_mod_config_.poly_type_,
                                                  -b, b));
                logic_shifts_.push_back(-shifts[i] / K);
                logic_constants_.push_back(constants[i]);
            }

            boot_context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Bootstrapping parameters is locked after "
                                     "generation and cannot be modified.");
        }

        cudaDeviceSynchronize();
    }

    __host__ Ciphertext<Scheme::CKKS>
    HELogicOperator<Scheme::CKKS>::bit_bootstrapping(
        Ciphertext<Scheme::CKKS>& input1, Galoiskey<Scheme::CKKS>& galois_key,
        Relinkey<Scheme::CKKS>& relin_key,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        input1.validate_context(context_);

        if (logic_polys_.size() != 1)
        {
            throw std::invalid_argument(
                "Bit bootstrapping parameters are not generated!");
        }

        return logic_bootstrapping(input1, 0, galois_key, relin_key,
                                   swk_dense_to_sparse, swk_sparse_to_dense,
                                   options);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HELogicOperator<Scheme::CKKS>::gate_bootstrapping(
        logic_gate gate_type, Ciphertext<Scheme::CKKS>& input1,
        Ciphertext<Scheme::CKKS>& input2, Galoiskey<Scheme::CKKS>& galois_key,
        Relinkey<Scheme::CKKS>& relin_key,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        if (logic_polys_.size() != 6)
        {
            throw std::invalid_argument(
                "Gate bootstrapping parameters are not generated!");
        }

        // Inputs above the SlotToCoeff start level are brought down to it.
        int level = stc_config_.level_start_;
        if (input1.level() > level || input2.level() > level)
        {
            Ciphertext<Scheme::CKKS> dropped1 = input1;
            Ciphertext<Scheme::CKKS> dropped2 = input2;
            while (dropped1.level() > level)
            {
                mod_drop_inplace(dropped1, options);
            }
            while (dropped2.level() > level)
            {
                mod_drop_inplace(dropped2, options);
            }
            return gate_bootstrapping(gate_type, dropped1, dropped2, galois_key,
                                      relin_key, swk_dense_to_sparse,
                                      swk_sparse_to_dense, options);
        }

        for (Ciphertext<Scheme::CKKS>* input : {&input1, &input2})
        {
            int current_decomp_count = context_->Q_size - input->depth_;
            if (current_decomp_count != (level + 1))
            {
                throw LevelMismatchError(error_message(
                    "Ciphertexts leveled should be at max! The ciphertext has ",
                    current_decomp_count, " moduli, expected ", level + 1,
                    "."));
            }
        }

        Ciphertext<Scheme::CKKS> input_ =
            operator_ciphertext(0, options.stream_);
        add(input1, input2, input_, options);

        return logic_bootstrapping(input_, static_cast<int>(gate_type),
                                   galois_key, relin_key, swk_dense_to_sparse,
                                   swk_sparse_to_dense, options);
    }

    __host__ Ciphertext<Scheme::CKKS>
    HELogicOperator<Scheme::CKKS>::logic_bootstrapping(
        Ciphertext<Scheme::CKKS>& input1, int function_index,
        Galoiskey<Scheme::CKKS>& galois_key, Relinkey<Scheme::CKKS>& relin_key,
        Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
        Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
        const ExecutionOptions& options)
    {
        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // An input above the SlotToCoeff start level is brought down to it.
        if (input1.level() > stc_config_.level_start_)
        {
            Ciphertext<Scheme::CKKS> dropped = input1;
            while (dropped.level() > stc_config_.level_start_)
            {
                mod_drop_inplace(dropped, options);
            }
            return logic_bootstrapping(dropped, function_index, galois_key,
                                       relin_key, swk_dense_to_sparse,
                                       swk_sparse_to_dense, options);
        }

        int current_decomp_count = context_->Q_size - input1.depth_;
        if (current_decomp_count != (stc_config_.level_start_ + 1))
        {
            throw LevelMismatchError(error_message(
                "Ciphertexts leveled should be at max! The ciphertext has ",
                current_decomp_count, " moduli, expected ",
                stc_config_.level_start_ + 1, "."));
        }

        // Slot to coeff
        Ciphertext<Scheme::CKKS> StoC_results =
            solo_slot_to_coeff(input1, galois_key, options);

        // Raise modulus
        Ciphertext<Scheme::CKKS> scaled_input1 = StoC_results;
        scale_up(
            StoC_results,
            std::round((static_cast<double>(context_->prime_vector_[0].value) /
                        eval_mod_config_.message_ratio_) /
                       StoC_results.scale()),
            scaled_input1, options);

        Ciphertext<Scheme::CKKS> c_raised = mod_up_from_q0(
            scaled_input1, swk_dense_to_sparse, swk_sparse_to_dense, options);
        c_raised.encoding_ = encoding::COEFFICIENT;

        // Coeff to slot
        Ciphertext<Scheme::CKKS> CtoS_results =
            solo_coeff_to_slot(c_raised, galois_key, options);

        Ciphertext<Scheme::CKKS> result =
            eval_mod(CtoS_results, logic_polys_[function_index],
                     logic_shifts_[function_index], relin_key, options);
        result.scale_ = scale_boot_;
        add_plain_complex(result,
                          Complex64(logic_constants_[function_index], 0.0),
                          result, options);
        result.encoding_ = encoding::SLOT;

        return result;
    }

    __host__ void HELogicOperator<Scheme::CKKS>::one_minus_cipher(
        Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
        const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        int current_decomp_count = context_->Q_size - input1.depth_;

        addition_plain_ckks_poly<<<dim3((context_->n >> 8),
                                        current_decomp_count, 2),
                                   256, 0, options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), output.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HELogicOperator<Scheme::CKKS>::one_minus_cipher_inplace(
        Ciphertext<Scheme::CKKS>& input1, const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        int current_decomp_count = context_->Q_size - input1.depth_;

        addition<<<dim3((context_->n >> 8), current_decomp_count, 1), 256, 0,
                   options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), input1.data(),
            context_->modulus_->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
