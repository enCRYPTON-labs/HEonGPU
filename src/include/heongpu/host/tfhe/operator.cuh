// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_OPERATOR_H
#define HEONGPU_TFHE_OPERATOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpufft/fft.cuh"
#include <heongpu/kernel/addition.cuh>
#include <heongpu/kernel/multiplication.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/bootstrapping.cuh>

#include <heongpu/host/tfhe/context.cuh>
#include <heongpu/host/tfhe/ciphertext.cuh>
#include <heongpu/host/tfhe/evaluationkey.cuh>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace heongpu
{

    template <> class HELogicOperator<Scheme::TFHE>
    {
      public:
        /**
         * @brief Initializes a logic operator for the TFHE scheme.
         *
         * Sets up logic gate evaluation using the given encryption context.
         *
         * @param context TFHE encryption context.
         */
        HELogicOperator(HEContext<Scheme::TFHE> context);

        /**
         * @brief Evaluates the NAND gate on two TFHE ciphertexts.
         *
         * Computes the NAND of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after NAND operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void NAND(Ciphertext<Scheme::TFHE>& input1,
                           Ciphertext<Scheme::TFHE>& input2,
                           Ciphertext<Scheme::TFHE>& output,
                           Bootstrappingkey<Scheme::TFHE>& boot_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            NAND_pre_computation(
                                                input1, input2, temp_cipher,
                                                options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the AND gate on two TFHE ciphertexts.
         *
         * Computes the AND of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after AND operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void AND(Ciphertext<Scheme::TFHE>& input1,
                          Ciphertext<Scheme::TFHE>& input2,
                          Ciphertext<Scheme::TFHE>& output,
                          Bootstrappingkey<Scheme::TFHE>& boot_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            AND_pre_computation(
                                                input1, input2, temp_cipher,
                                                options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the NOR gate on two TFHE ciphertexts.
         *
         * Computes the NOR of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after NOR operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void NOR(Ciphertext<Scheme::TFHE>& input1,
                          Ciphertext<Scheme::TFHE>& input2,
                          Ciphertext<Scheme::TFHE>& output,
                          Bootstrappingkey<Scheme::TFHE>& boot_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            NOR_pre_computation(
                                                input1, input2, temp_cipher,
                                                options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the OR gate on two TFHE ciphertexts.
         *
         * Computes the OR of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after OR operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void OR(Ciphertext<Scheme::TFHE>& input1,
                         Ciphertext<Scheme::TFHE>& input2,
                         Ciphertext<Scheme::TFHE>& output,
                         Bootstrappingkey<Scheme::TFHE>& boot_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            OR_pre_computation(input1, input2,
                                                               temp_cipher,
                                                               options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the XNOR gate on two TFHE ciphertexts.
         *
         * Computes the XNOR of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after XNOR operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void XNOR(Ciphertext<Scheme::TFHE>& input1,
                           Ciphertext<Scheme::TFHE>& input2,
                           Ciphertext<Scheme::TFHE>& output,
                           Bootstrappingkey<Scheme::TFHE>& boot_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            XNOR_pre_computation(
                                                input1, input2, temp_cipher,
                                                options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the XOR gate on two TFHE ciphertexts.
         *
         * Computes the XOR of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after XOR operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void XOR(Ciphertext<Scheme::TFHE>& input1,
                          Ciphertext<Scheme::TFHE>& input2,
                          Ciphertext<Scheme::TFHE>& output,
                          Bootstrappingkey<Scheme::TFHE>& boot_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.shape_ != input2.shape_)
            {
                throw std::runtime_error(
                    "Both ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            XOR_pre_computation(
                                                input1, input2, temp_cipher,
                                                options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher, temp_cipher2,
                                                boot_key, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher2, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the NOT gate on two TFHE ciphertexts.
         *
         * Computes the NOT of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after NOT operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void NOT(Ciphertext<Scheme::TFHE>& input1,
                          Ciphertext<Scheme::TFHE>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (!(input1.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::TFHE>& output_)
                        {
                            output.a_device_location_.resize(
                                input1.shape_ * context_->n_, options.stream_);
                            output.b_device_location_.resize(input1.shape_,
                                                             options.stream_);
                            output.n_ = input1.n_;
                            output.shape_ = input1.shape_;
                            output.variances_ = input1.variances_;
                            output.alpha_min_ = input1.alpha_min_;
                            output.alpha_max_ = input1.alpha_max_;
                            output.ciphertext_generated_ = true;
                            output.storage_type_ = storage_type::DEVICE;

                            NOT_computation(input1, output, options.stream_);
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Evaluates the MUX(multiplexer) gate on two TFHE ciphertexts.
         *
         * Computes the MUX of `input1` and `input2`, writing the result to
         * `output`. Requires the bootstrapping key for ciphertext refresh.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext after MUX operation.
         * @param boot_key Bootstrapping key for the operation.
         * @param options Optional CUDA execution settings.
         */
        __host__ void MUX(Ciphertext<Scheme::TFHE>& input1,
                          Ciphertext<Scheme::TFHE>& input2,
                          Ciphertext<Scheme::TFHE>& control_input,
                          Ciphertext<Scheme::TFHE>& output,
                          Bootstrappingkey<Scheme::TFHE>& boot_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if ((input1.shape_ != input2.shape_) ||
                (input1.shape_ != control_input.shape_))
            {
                throw std::runtime_error("Ciphertexts size should be equal!");
            }

            if (!(input1.ciphertext_generated_ && input2.ciphertext_generated_))
            {
                throw std::runtime_error(
                    "One or both inputs are not generated! Encrypt both "
                    "ciphertexts before using them.");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::TFHE>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::TFHE>& input2_)
                        {
                            input_storage_manager(
                                boot_key,
                                [&](Bootstrappingkey<Scheme::TFHE>& boot_key_)
                                {
                                    output_storage_manager(
                                        output,
                                        [&](Ciphertext<Scheme::TFHE>& output_)
                                        {
                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher1 =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            AND_pre_computation(
                                                control_input, input1,
                                                temp_cipher1, options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher2 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher1, temp_cipher2,
                                                boot_key, options.stream_);

                                            //

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher3 =
                                                    generate_empty_ciphertext(
                                                        context_->n_,
                                                        input1.shape_,
                                                        options.stream_);

                                            AND_N_pre_computation(
                                                control_input, input2,
                                                temp_cipher3, options.stream_);

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher4 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            bootstrapping(
                                                temp_cipher3, temp_cipher4,
                                                boot_key, options.stream_);

                                            //

                                            Ciphertext<Scheme::TFHE>
                                                temp_cipher5 =
                                                    generate_empty_ciphertext(
                                                        (context_->k_ *
                                                         context_->N_),
                                                        input1.shape_,
                                                        options.stream_);

                                            OR_pre_computation(
                                                temp_cipher2, temp_cipher4,
                                                temp_cipher5, options.stream_);

                                            output.a_device_location_.resize(
                                                input1.shape_ * context_->n_,
                                                options.stream_);
                                            output.b_device_location_.resize(
                                                input1.shape_, options.stream_);
                                            output.n_ = input1.n_;
                                            output.shape_ = input1.shape_;
                                            output.variances_ =
                                                input1.variances_;
                                            output.alpha_min_ =
                                                input1.alpha_min_;
                                            output.alpha_max_ =
                                                input1.alpha_max_;
                                            output.ciphertext_generated_ = true;
                                            output.storage_type_ =
                                                storage_type::DEVICE;

                                            key_switching(temp_cipher5, output,
                                                          boot_key,
                                                          options.stream_);
                                        },
                                        options);
                                },
                                options, false);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

      private:
        __host__ void NAND_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                           Ciphertext<Scheme::TFHE>& input2,
                                           Ciphertext<Scheme::TFHE>& output,
                                           cudaStream_t stream);

        __host__ void AND_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                          Ciphertext<Scheme::TFHE>& input2,
                                          Ciphertext<Scheme::TFHE>& output,
                                          cudaStream_t stream);

        __host__ void AND_N_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                            Ciphertext<Scheme::TFHE>& input2,
                                            Ciphertext<Scheme::TFHE>& output,
                                            cudaStream_t stream);

        __host__ void NOR_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                          Ciphertext<Scheme::TFHE>& input2,
                                          Ciphertext<Scheme::TFHE>& output,
                                          cudaStream_t stream);

        __host__ void OR_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                         Ciphertext<Scheme::TFHE>& input2,
                                         Ciphertext<Scheme::TFHE>& output,
                                         cudaStream_t stream);

        __host__ void XNOR_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                           Ciphertext<Scheme::TFHE>& input2,
                                           Ciphertext<Scheme::TFHE>& output,
                                           cudaStream_t stream);

        __host__ void XOR_pre_computation(Ciphertext<Scheme::TFHE>& input1,
                                          Ciphertext<Scheme::TFHE>& input2,
                                          Ciphertext<Scheme::TFHE>& output,
                                          cudaStream_t stream);

        __host__ void NOT_computation(Ciphertext<Scheme::TFHE>& input1,
                                      Ciphertext<Scheme::TFHE>& output,
                                      cudaStream_t stream);

        __host__ void bootstrapping(Ciphertext<Scheme::TFHE>& input,
                                    Ciphertext<Scheme::TFHE>& output,
                                    Bootstrappingkey<Scheme::TFHE>& boot_key,
                                    cudaStream_t stream);

        __host__ void key_switching(Ciphertext<Scheme::TFHE>& input,
                                    Ciphertext<Scheme::TFHE>& output,
                                    Bootstrappingkey<Scheme::TFHE>& boot_key,
                                    cudaStream_t stream);

        __host__ Ciphertext<Scheme::TFHE>
        generate_empty_ciphertext(int n, int shape, cudaStream_t stream);

        __host__ int32_t encode_to_torus32(uint32_t mu, uint32_t m_size);

      private:
        HEContext<Scheme::TFHE> context_;

        int Npower_;
        int32_t encode_mu;
    };

} // namespace heongpu

#endif // HEONGPU_TFHE_OPERATOR_H
