// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_OPERATOR_H
#define HEONGPU_CKKS_OPERATOR_H

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <functional>
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpufft/fft.cuh"
#include <heongpu/kernel/addition.cuh>
#include <heongpu/kernel/multiplication.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/bootstrapping.cuh>

#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include <heongpu/host/ckks/bootstrapping.cuh>

namespace heongpu
{
    struct CKKSEncodingTransformConfig
    {
        int CtoS_piece_; // [2, 5]
        int StoC_piece_; // [2, 5]
        int CtoS_start_level_; // ciphertext level where CoeffToSlot starts
        int StoC_start_level_; // ciphertext level where SlotToCoeff starts

        CKKSEncodingTransformConfig(int CtoS_piece = 3, int StoC_piece = 3)
            : CtoS_piece_(CtoS_piece), StoC_piece_(StoC_piece),
              CtoS_start_level_(-1), StoC_start_level_(-1)
        {
        }

        CKKSEncodingTransformConfig(int CtoS_piece, int StoC_piece,
                                    int CtoS_start_level, int StoC_start_level)
            : CtoS_piece_(CtoS_piece), StoC_piece_(StoC_piece),
              CtoS_start_level_(CtoS_start_level),
              StoC_start_level_(StoC_start_level)
        {
        }
    };

    struct CKKSEncodingTransformContext
    {
        bool generated_ = false;

        int CtoS_piece_ = 0;
        int StoC_piece_ = 0;
        int CtoS_level_ = 0;
        int StoC_level_ = 0;

        std::vector<int> key_indexs_;

        std::vector<heongpu::DeviceVector<Data64>> V_matrixs_rotated_encoded_;
        std::vector<heongpu::DeviceVector<Data64>>
            V_inv_matrixs_rotated_encoded_;

        std::vector<std::vector<std::vector<int>>> diags_matrices_bsgs_;
        std::vector<std::vector<std::vector<int>>> diags_matrices_inv_bsgs_;

        std::vector<std::vector<int>> diags_matrices_bsgs_rot_n1_;
        std::vector<std::vector<int>> diags_matrices_bsgs_rot_n2_;

        std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n1_;
        std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n2_;
    };

    /**
     * @brief HEOperator is responsible for performing homomorphic operations on
     * encrypted data, such as addition, subtraction, multiplication, and other
     * functions.
     *
     * The HEOperator class is initialized with encryption parameters and
     * provides various functions for performing operations on ciphertexts,
     * including CKKS scheme. It supports both in-place and
     * out-of-place operations, as well as asynchronous processing using CUDA
     * streams.
     */
    template <> class HEOperator<Scheme::CKKS>
    {
      protected:
        /**
         * @brief Construct a new HEOperator object with the given parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters for the operator.
         */
        __host__ HEOperator(HEContext<Scheme::CKKS> context,
                            HEEncoder<Scheme::CKKS>& encoder);

      public:
        /**
         * @brief Adds two ciphertexts and stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Adds the second ciphertext to the first ciphertext, modifying
         * the first ciphertext with the result.
         *
         * @param input1 The ciphertext to which the value of input2 will be
         * added.
         * @param input2 The ciphertext to be added to input1.
         */
        __host__ void
        add_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            add(input1, input2, input1, options);
        }

        /**
         * @brief Subtracts the second ciphertext from the first and stores the
         * result in the output.
         *
         * @param input1 First input ciphertext (minuend).
         * @param input2 Second input ciphertext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         */
        __host__ void sub(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Subtracts the second ciphertext from the first, modifying the
         * first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         */
        __host__ void
        sub_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            sub(input1, input2, input1, options);
        }

        /**
         * @brief Negates a ciphertext and stores the result in the output.
         *
         * @param input1 Input ciphertext to be negated.
         * @param output Ciphertext where the result of the negation is stored.
         */
        __host__ void
        negate(Ciphertext<Scheme::CKKS>& input1,
               Ciphertext<Scheme::CKKS>& output,
               const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Negates a ciphertext in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         */
        __host__ void
        negate_inplace(Ciphertext<Scheme::CKKS>& input1,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            negate(input1, input1, options);
        }

        /**
         * @brief Adds a ciphertext and a plaintext and stores the result in the
         * output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input plaintext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void
        add_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Plaintext<Scheme::CKKS>& second)
                              { add_plain(first, second, output, options); }))
            {
                return;
            }

            input2.validate_context(context_);

            input1.validate_context(context_);

            if (input1.depth_ != input2.depth_)
            {
                throw LevelMismatchError(
                    error_message("Ciphertexts leveled are not equal (depth ",
                                  input1.depth_, " and ", input2.depth_,
                                  ")! Use mod_drop to bring both operands to "
                                  "the same depth."));
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext should have same encoding type");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    add_plain_ckks(input1_, input2_, output_,
                                                   options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.encoding_ = input1_.encoding_;
                                    output_.scale_ = input1_.scale_;
                                    output_.rescale_required_ =
                                        input1_.rescale_required_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                    output_.ciphertext_generated_ = true;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Adds a plaintext to a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Plaintext to be added to the ciphertext.
         */
        __host__ void
        add_plain_inplace(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Plaintext<Scheme::CKKS>& second)
                              {
                                  add_plain_inplace(first, second, options);
                                  if (&first != &input1)
                                  {
                                      input1 = std::move(first);
                                  }
                              }))
            {
                return;
            }

            input2.validate_context(context_);

            input1.validate_context(context_);

            if (input1.depth_ != input2.depth_)
            {
                throw LevelMismatchError(
                    error_message("Ciphertexts leveled are not equal (depth ",
                                  input1.depth_, " and ", input2.depth_,
                                  ")! Use mod_drop to bring both operands to "
                                  "the same depth."));
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext should have same encoding type");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_) {
                            add_plain_ckks_inplace(input1_, input2_,
                                                   options.stream_);
                        },
                        options, false);
                },
                options, true);
        }

        //

        /**
         * @brief Adds a ciphertext and a plaintext and stores the result in the
         * output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input constant plaintext(double) to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void
        add_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            add_constant_plain_ckks(input1_,
                                                    input2 * input1_.scale_,
                                                    output_, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Adds a plaintext to a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Input constant plaintext(double) to be added.
         */
        __host__ void
        add_plain_inplace(Ciphertext<Scheme::CKKS>& input1, double input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    add_constant_plain_ckks_inplace(
                        input1_, input2 * input1_.scale_, options.stream_);
                },
                options, true);
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext (minuend).
         * @param input2 Input plaintext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         */
        __host__ void
        sub_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Plaintext<Scheme::CKKS>& second)
                              { sub_plain(first, second, output, options); }))
            {
                return;
            }

            input2.validate_context(context_);

            input1.validate_context(context_);

            if (input1.depth_ != input2.depth_)
            {
                throw LevelMismatchError(
                    error_message("Ciphertexts leveled are not equal (depth ",
                                  input1.depth_, " and ", input2.depth_,
                                  ")! Use mod_drop to bring both operands to "
                                  "the same depth."));
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext should have same encoding type");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    sub_plain_ckks(input1_, input2_, output_,
                                                   options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.encoding_ = input1_.encoding_;
                                    output_.scale_ = input1_.scale_;
                                    output_.rescale_required_ =
                                        input1_.rescale_required_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                    output_.ciphertext_generated_ = true;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext from which the plaintext will be subtracted.
         * @param input2 Plaintext to be subtracted from the ciphertext.
         */
        __host__ void
        sub_plain_inplace(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Plaintext<Scheme::CKKS>& second)
                              {
                                  sub_plain_inplace(first, second, options);
                                  if (&first != &input1)
                                  {
                                      input1 = std::move(first);
                                  }
                              }))
            {
                return;
            }

            input2.validate_context(context_);

            input1.validate_context(context_);

            if (input1.depth_ != input2.depth_)
            {
                throw LevelMismatchError(
                    error_message("Ciphertexts leveled are not equal (depth ",
                                  input1.depth_, " and ", input2.depth_,
                                  ")! Use mod_drop to bring both operands to "
                                  "the same depth."));
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext should have same encoding type");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_) {
                            sub_plain_ckks_inplace(input1_, input2_,
                                                   options.stream_);
                        },
                        options, false);
                },
                options, true);
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext (minuend).
         * @param input2 Input plaintext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         */
        __host__ void
        sub_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            sub_constant_plain_ckks(input1_,
                                                    input2 * input1_.scale_,
                                                    output_, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext from which the plaintext will be subtracted.
         * @param input2 Plaintext to be subtracted from the ciphertext.
         */
        __host__ void
        sub_plain_inplace(Ciphertext<Scheme::CKKS>& input1, double input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    sub_constant_plain_ckks_inplace(
                        input1_, input2 * input1_.scale_, options.stream_);
                },
                options, true);
        }

        /**
         * @brief Adds a complex constant to a ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be added.
         * @param c Complex constant to be added (not multiplied by scale).
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void
        add_plain_complex(Ciphertext<Scheme::CKKS>& input1, Complex64 c,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            add_constant_plain_complex_ckks(input1_, c, output_,
                                                            options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ =
                                input1_.relinearization_required_ ? 3 : 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies two ciphertexts and stores the result in the
         * output.
         *
         * @param input1 First input ciphertext to be multiplied.
         * @param input2 Second input ciphertext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void
        multiply(Ciphertext<Scheme::CKKS>& input1,
                 Ciphertext<Scheme::CKKS>& input2,
                 Ciphertext<Scheme::CKKS>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, false, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Ciphertext<Scheme::CKKS>& second)
                              { multiply(first, second, output, options); }))
            {
                return;
            }

            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_ ||
                input2.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the "
                    "non-linear part! Please use relinearization operation!");
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Both Ciphertexts should have same encoding type");
            }

            if (input1.rescale_required_ || input2.rescale_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the noise! "
                    "Please use rescale operation to get rid of additional "
                    "noise!");
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
                                    multiply_ckks(input1_, input2_, output_,
                                                  options.stream_);
                                    output_.rescale_required_ = true;

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 3;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.encoding_ = input1_.encoding_;
                                    output_.relinearization_required_ = true;
                                    output_.ciphertext_generated_ = true;
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies two ciphertexts in-place, modifying the first
         * ciphertext.
         *
         * @param input1 Ciphertext to be multiplied, and where the result will
         * be stored.
         * @param input2 Second input ciphertext to be multiplied.
         */
        __host__ void
        multiply_inplace(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            multiply(input1, input2, input1, options);
        }

        /**
         * @brief Multiplies a ciphertext and a plaintext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param input2 Input plaintext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void
        multiply_plain(Ciphertext<Scheme::CKKS>& input1,
                       Plaintext<Scheme::CKKS>& input2,
                       Ciphertext<Scheme::CKKS>& output,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, input2, options,
                              [&](Ciphertext<Scheme::CKKS>& first,
                                  Plaintext<Scheme::CKKS>& second) {
                                  multiply_plain(first, second, output,
                                                 options);
                              }))
            {
                return;
            }

            input2.validate_context(context_);

            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            if (input1.encoding_ != input2.encoding_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext should have same encoding type");
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
                    " coefficients, got ", input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    if (input2_.size() <
                                        (context_->n * current_decomp_count))
                                    {
                                        throw ParameterError(error_message(
                                            "Invalid Plaintext size! Expected "
                                            "at least ",
                                            (context_->n *
                                             current_decomp_count),
                                            " coefficients, got ",
                                            input2_.size(), "."));
                                    }

                                    multiply_plain_ckks(input1_, input2_,
                                                        output_,
                                                        options.stream_);
                                    output_.rescale_required_ = true;

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.encoding_ = input1_.encoding_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                    output_.ciphertext_generated_ = true;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Plaintext to be multiplied with the ciphertext.
         */
        __host__ void multiply_plain_inplace(
            Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            multiply_plain(input1, input2, input1, options);
        }

        /**
         * @brief Multiplies a ciphertext and a plaintext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param input2 Input constant plaintext(double) to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void
        multiply_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                       Ciphertext<Scheme::CKKS>& output, double scale,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            // The constant gets the scale of the level, so that the
            // rescaled product has the scale of the next level.
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first)
                              {
                                  check_level_scale(first);
                                  multiply_plain(
                                      first, input2, output,
                                      context_->level_scales_[first.level()],
                                      options);
                              }))
            {
                return;
            }

            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
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
                            multiply_const_plain_ckks(input1_, input2, output_,
                                                      scale, options.stream_);
                            output_.rescale_required_ = true;

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Input constant plaintext(double) to be multiplied.
         */
        __host__ void multiply_plain_inplace(
            Ciphertext<Scheme::CKKS>& input1, double input2, double scale,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            multiply_plain(input1, input2, input1, scale, options);
        }

        /**
         * @brief Multiplies a ciphertext by a complex constant and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param c Complex constant to multiply with the ciphertext.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void multiply_plain_complex(
            Ciphertext<Scheme::CKKS>& input1, Complex64 c,
            Ciphertext<Scheme::CKKS>& output,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            multiply_const_plain_complex_ckks(
                                input1_, c, output_, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ =
                                input1_.relinearization_required_ ? 3 : 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Scales up a ciphertext by multiplying its scale factor.
         *
         * @param input Input ciphertext to be scaled up.
         * @param scale Scaling factor to multiply with the ciphertext's current
         * scale.
         * @param output Ciphertext where the scaled result is stored.
         */
        __host__ void
        scale_up(Ciphertext<Scheme::CKKS>& input, double scale,
                 Ciphertext<Scheme::CKKS>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            input.validate_context(context_);

            input_storage_manager(
                input,
                [&](Ciphertext<Scheme::CKKS>& input_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            scale_up_ckks(input_, scale, output_,
                                          options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ =
                                input_.relinearization_required_ ? 3 : 2;
                            output_.depth_ = input_.depth_;
                            output_.in_ntt_domain_ = input_.in_ntt_domain_;
                            output_.encoding_ = input_.encoding_;
                            output_.rescale_required_ =
                                input_.rescale_required_;
                            output_.relinearization_required_ =
                                input_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input == &output));
        }

        /**
         * @brief Multiplies a ciphertext by the imaginary unit i and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be multiplied by i.
         * @param output Ciphertext where the result is stored.
         */
        __host__ void
        mult_i(Ciphertext<Scheme::CKKS>& input1,
               Ciphertext<Scheme::CKKS>& output,
               const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            mult_i_ckks(input1_, output_, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ =
                                input1_.relinearization_required_ ? 3 : 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Divides a ciphertext by the imaginary unit i and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be divided by i.
         * @param output Ciphertext where the result is stored.
         */
        __host__ void
        div_i(Ciphertext<Scheme::CKKS>& input1,
              Ciphertext<Scheme::CKKS>& output,
              const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            div_i_ckks(input1_, output_, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ =
                                input1_.relinearization_required_ ? 3 : 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         */
        __host__ void relinearize_inplace(
            Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);

            if ((!input1.relinearization_required_))
            {
                throw std::invalid_argument(
                    "Ciphertexts can not use relinearization, since no "
                    "non-linear part!");
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (3 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (3 * context_->n * current_decomp_count),
                    " coefficients, got ", input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    switch (static_cast<int>(relin_key.key_type))
                    {
                        case 2: // KEYSWITCHING_METHOD_II
                            relinearize_external_product_method2_inplace_ckks(
                                input1_, relin_key, options.stream_);
                            break;
                        default:
                            throw ParameterError(error_message(
                                "Invalid Key Switching Type ",
                                static_cast<int>(relin_key.key_type),
                                "! Expected KEYSWITCHING_METHOD_II (2)."));
                            break;
                    }

                    input1_.cipher_size_ = 2;
                    input1_.relinearization_required_ = false;
                },
                options, true);
        }

        /**
         * @brief Rotates the rows of a ciphertext by a given shift value and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         */
        __host__ void
        rotate_rows(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& output,
                    Galoiskey<Scheme::CKKS>& galois_key, int shift,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first) {
                                  rotate_rows(first, output, galois_key, shift,
                                              options);
                              }))
            {
                return;
            }

            galois_key.validate_context(context_);

            input1.validate_context(context_);

            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw StateError(error_message(
                    "Ciphertext can not be rotated! ",
                    input1.relinearization_required_ ? "Relinearize it first."
                                                     : "Rescale it first."));
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
                    " coefficients, got ", input1.memory_size(), "."));
            }

            // Rotating by a multiple of the row size gives the input.
            shift %= (context_->n >> 1);
            if (shift == 0)
            {
                output = input1;
                return;
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 2: // KEYSWITCHING_METHOD_II
                                    rotate_ckks_method_II(input1_, output_,
                                                          galois_key, shift,
                                                          options.stream_);
                                    break;
                                default:
                                    throw ParameterError(error_message(
                                        "Invalid Key Switching Type ",
                                        static_cast<int>(galois_key.key_type),
                                        "! Expected KEYSWITCHING_METHOD_II "
                                        "(2)."));
                                    break;
                            }
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Rotates the rows of a ciphertext in-place by a given shift
         * value, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be rotated.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         */
        __host__ void rotate_rows_inplace(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key, int shift,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);

            input1.validate_context(context_);

            if (shift == 0)
            {
                return;
            }

            rotate_rows(input1, input1, galois_key, shift, options);
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to which the Galois operation will be
         * applied.
         * @param output Ciphertext where the result of the Galois operation is
         * stored.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         */
        __host__ void
        apply_galois(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& output,
                     Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first) {
                                  apply_galois(first, output, galois_key,
                                               galois_elt, options);
                              }))
            {
                return;
            }

            galois_key.validate_context(context_);

            input1.validate_context(context_);

            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw StateError(error_message(
                    "Ciphertext can not be rotated! ",
                    input1.relinearization_required_ ? "Relinearize it first."
                                                     : "Rescale it first."));
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
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
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 2: // KEYSWITCHING_METHOD_II
                                    apply_galois_ckks_method_II(
                                        input1_, output_, galois_key,
                                        galois_elt, options.stream_);
                                    break;
                                default:
                                    throw ParameterError(error_message(
                                        "Invalid Key Switching Type ",
                                        static_cast<int>(galois_key.key_type),
                                        "! Expected KEYSWITCHING_METHOD_II "
                                        "(2)."));
                                    break;
                            }

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to which the Galois operation will be
         * applied.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         */
        __host__ void apply_galois_inplace(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);

            input1.validate_context(context_);

            apply_galois(input1, input1, galois_key, galois_elt, options);
        }

        /**
         * @brief Performs key switching on the ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be key-switched.
         * @param output Ciphertext where the result of the key switching is
         * stored.
         * @param switch_key Switch key used for the key switching operation.
         */
        __host__ void
        keyswitch(Ciphertext<Scheme::CKKS>& input1,
                  Ciphertext<Scheme::CKKS>& output,
                  Switchkey<Scheme::CKKS>& switch_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first) {
                                  keyswitch(first, output, switch_key, options);
                              }))
            {
                return;
            }

            switch_key.validate_context(context_);

            input1.validate_context(context_);

            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw StateError(error_message(
                    "Ciphertext can not be rotated! ",
                    input1.relinearization_required_ ? "Relinearize it first."
                                                     : "Rescale it first."));
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
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
                            switch (static_cast<int>(switch_key.key_type))
                            {
                                case 2: // KEYSWITCHING_METHOD_II
                                    switchkey_ckks_method_II(input1_, output_,
                                                             switch_key,
                                                             options.stream_);
                                    break;
                                default:
                                    throw ParameterError(error_message(
                                        "Invalid Key Switching Type ",
                                        static_cast<int>(switch_key.key_type),
                                        "! Expected KEYSWITCHING_METHOD_II "
                                        "(2)."));
                                    break;
                            }

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs conjugation on the ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be conjugated.
         * @param output Ciphertext where the result of the conjugation is
         * stored.
         * @param conjugate_key Switch key used for the conjugation operation.
         */
        __host__ void
        conjugate(Ciphertext<Scheme::CKKS>& input1,
                  Ciphertext<Scheme::CKKS>& output,
                  Galoiskey<Scheme::CKKS>& conjugate_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first) {
                                  conjugate(first, output, conjugate_key,
                                            options);
                              }))
            {
                return;
            }

            conjugate_key.validate_context(context_);

            input1.validate_context(context_);

            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw StateError(error_message(
                    "Ciphertext can not be rotated! ",
                    input1.relinearization_required_ ? "Relinearize it first."
                                                     : "Rescale it first."));
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
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
                            switch (static_cast<int>(conjugate_key.key_type))
                            {
                                case 2: // KEYSWITHING_METHOD_II
                                    conjugate_ckks_method_II(input1_, output_,
                                                             conjugate_key,
                                                             options.stream_);
                                    break;
                                default:
                                    throw ParameterError(error_message(
                                        "Invalid Key Switching Type ",
                                        static_cast<int>(
                                            conjugate_key.key_type),
                                        "! Expected KEYSWITCHING_METHOD_II "
                                        "(2)."));
                                    break;
                            }

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.encoding_ = input1_.encoding_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Rescales a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to be rescaled.
         */
        __host__ void
        rescale_inplace(Ciphertext<Scheme::CKKS>& input1,
                        const ExecutionOptions& options = ExecutionOptions())
        {
            // The automatic mode rescales when needed; code written for the
            // manual mode can rescale anyway.
            if (automatic() && !input1.rescale_required_)
            {
                return;
            }

            input1.validate_context(context_);

            if ((!input1.rescale_required_) || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertexts can not be rescaled!");
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (current_decomp_count <= 1)
            {
                throw std::logic_error(
                    "Ciphertext modulus can not be rescaled!");
            }

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
                    " coefficients, got ", input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                { rescale_inplace_ckks_leveled(input1_, options.stream_); },
                options, true);

            input1.rescale_required_ = false;
        }

        /**
         * @brief Drop the last modulus of ciphertext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input ciphertext from which last modulus will be
         * dropped.
         * @param output Ciphertext where the result of the modulus drop is
         * stored.
         */
        __host__ void
        mod_drop(Ciphertext<Scheme::CKKS>& input1,
                 Ciphertext<Scheme::CKKS>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            // The automatic mode keeps the scale of the level.
            if (run_automatic(input1, options,
                              [&](Ciphertext<Scheme::CKKS>& first)
                              {
                                  Ciphertext<Scheme::CKKS> storage;
                                  output = dropped(first, first.depth_ + 1,
                                                   storage, options);
                              }))
            {
                return;
            }

            input1.validate_context(context_);

            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext's modulus can not be dropped!!");
            }

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
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
                            mod_drop_ckks_leveled(input1_, output_,
                                                  options.stream_);

                            output.scheme_ = context_->scheme_;
                            output.context_id_ = context_->context_id_;
                            output.ring_size_ = context_->n;
                            output.coeff_modulus_count_ = context_->Q_size;
                            output.cipher_size_ = 2;
                            output.depth_ = input1.depth_ + 1;
                            output.scale_ = input1.scale_;
                            output.in_ntt_domain_ = input1.in_ntt_domain_;
                            output.encoding_ = input1.encoding_;
                            output.rescale_required_ = input1.rescale_required_;
                            output.relinearization_required_ =
                                input1.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Drop the last modulus of plaintext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input plaintext from which modulus will be dropped.
         * @param output Plaintext where the result of the modulus drop is
         * stored.
         */
        __host__ void
        mod_drop(Plaintext<Scheme::CKKS>& input1,
                 Plaintext<Scheme::CKKS>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.size() < (context_->n * current_decomp_count))
            {
                throw ParameterError(
                    error_message("Invalid Plaintext size! Expected at least ",
                                  (context_->n * current_decomp_count),
                                  " coefficients, got ", input1.size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Plaintext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Plaintext<Scheme::CKKS>& output_)
                        {
                            mod_drop_ckks_plaintext(input1_, output_,
                                                    options.stream_);

                            output.scheme_ = input1.scheme_;
                            output.context_id_ = input1.context_id_;
                            output.plain_size_ =
                                (context_->n * (current_decomp_count - 1));
                            output.depth_ = input1.depth_ + 1;
                            output.scale_ = input1.scale_;
                            output.in_ntt_domain_ = input1.in_ntt_domain_;
                            output.encoding_ = input1.encoding_;
                            output.plaintext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Drop the last modulus of plaintext in-place on a plaintext,
         * modifying the input plaintext.
         *
         * @param input1 Plaintext to perform modulus dropping on.
         */
        __host__ void
        mod_drop_inplace(Plaintext<Scheme::CKKS>& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.size() < (context_->n * current_decomp_count))
            {
                throw ParameterError(
                    error_message("Invalid Plaintext size! Expected at least ",
                                  (context_->n * current_decomp_count),
                                  " coefficients, got ", input1.size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Plaintext<Scheme::CKKS>& input1_)
                { mod_drop_ckks_plaintext_inplace(input1_, options.stream_); },
                options, true);
        }

        /**
         * @brief Drop the last modulus of ciphertext in-place on a ciphertext,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to perform modulus dropping on.
         */
        __host__ void
        mod_drop_inplace(Ciphertext<Scheme::CKKS>& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            if (automatic())
            {
                Ciphertext<Scheme::CKKS> output;
                mod_drop(input1, output, options);
                input1 = std::move(output);
                return;
            }

            input1.validate_context(context_);

            int current_decomp_count = context_->Q_size - input1.depth_;

            if (input1.memory_size() < (2 * context_->n * current_decomp_count))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * current_decomp_count),
                    " coefficients, got ", input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                { mod_drop_ckks_leveled_inplace(input1_, options.stream_); },
                options, true);
        }

        HEOperator() = default;
        HEOperator(const HEOperator& copy) = default;
        HEOperator(HEOperator&& source) = default;
        HEOperator& operator=(const HEOperator& assign) = default;
        HEOperator& operator=(HEOperator&& assign) = default;

        /**
         * @brief Returns a ciphertext whose slots all hold the sum of the slots
         * of input1. Uses log2(slot count) rotations by the steps 1, 2, 4, ...,
         * slot count / 2. A Galoiskey(context, log2(slot count) - 1) has a key
         * for each of them; the default Galoiskey also works, with more
         * rotations for the larger steps.
         */
        __host__ Ciphertext<Scheme::CKKS>
        sum_slots(Ciphertext<Scheme::CKKS>& input1,
                  Galoiskey<Scheme::CKKS>& galois_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> result = input1;
            for (int step = 1; step < (context_->n >> 1); step <<= 1)
            {
                Ciphertext<Scheme::CKKS> rotated(context_, options);
                rotate_rows(result, rotated, galois_key, step, options);
                add_inplace(result, rotated, options);
            }
            return result;
        }

        /**
         * @brief Returns a ciphertext whose slots all hold the inner product
         * of the slots of input1 and input2. Multiplies, relinearizes and
         * rescales, so the result is one level lower than the inputs, and
         * then calls sum_slots.
         */
        __host__ Ciphertext<Scheme::CKKS>
        inner_product(Ciphertext<Scheme::CKKS>& input1,
                      Ciphertext<Scheme::CKKS>& input2,
                      Relinkey<Scheme::CKKS>& relin_key,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> product(context_, options);
            multiply(input1, input2, product, options);
            relinearize_inplace(product, relin_key, options);
            rescale_inplace(product, options);
            return sum_slots(product, galois_key, options);
        }

        /**
         * @brief Returns a ciphertext whose slots all hold the inner product
         * of the slots of input1 and input2. Multiplies and rescales, so the
         * result is one level lower than input1, and then calls sum_slots.
         */
        __host__ Ciphertext<Scheme::CKKS>
        inner_product(Ciphertext<Scheme::CKKS>& input1,
                      Plaintext<Scheme::CKKS>& input2,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> product(context_, options);
            multiply_plain(input1, input2, product, options);
            rescale_inplace(product, options);
            return sum_slots(product, galois_key, options);
        }

        /**
         * @brief Returns input1 rotated by each of shifts: the same results as
         * rotate_rows(input1, output, galois_key, shift) for each shift. The
         * rotations by shifts that galois_key has a key for share the
         * decomposition of input1 that key switching starts with, which is
         * computed once, so each of them after the first costs less than a
         * rotate_rows. Other shifts are composed from the power of two keys,
         * as rotate_rows does.
         */
        __host__ std::vector<Ciphertext<Scheme::CKKS>> rotate_rows_hoisted(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key, const std::vector<int>& shifts,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Returns the polynomial
         * coefficients[0] + coefficients[1] * x + ... + coefficients[d] * x^d
         * evaluated on the slots x of input1, with the same scale as input1.
         *
         * Uses ceil(log2(d + 1)) levels, where d = coefficients.size() - 1;
         * input1 must have at least as many levels left. The values of x
         * should stay small (about |x| <= 1) for the result to be precise;
         * for other ranges, evaluate_chebyshev is more stable.
         *
         * @throws ParameterError if the degree is less than 1.
         * @throws LevelMismatchError if input1 does not have enough levels.
         */
        __host__ Ciphertext<Scheme::CKKS> evaluate_polynomial(
            Ciphertext<Scheme::CKKS>& input1,
            const std::vector<double>& coefficients,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Returns the Chebyshev series
         * coefficients[0] * T0(y) + ... + coefficients[d] * Td(y), where
         * y = (2x - a - b) / (b - a), evaluated on the slots x of input1, with
         * the same scale as input1. The slots must lie in [a, b].
         *
         * Uses ceil(log2(d + 1)) levels, where d = coefficients.size() - 1,
         * and one more unless [a, b] is [-1, 1].
         *
         * @throws ParameterError if the degree is less than 1 or a >= b.
         * @throws LevelMismatchError if input1 does not have enough levels.
         */
        __host__ Ciphertext<Scheme::CKKS> evaluate_chebyshev(
            Ciphertext<Scheme::CKKS>& input1,
            const std::vector<double>& coefficients, double a, double b,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Approximates function on [a, b] by Chebyshev interpolation
         * of the given degree and evaluates the approximation on the slots of
         * input1; see chebyshev_coefficients and evaluate_chebyshev.
         */
        __host__ Ciphertext<Scheme::CKKS> evaluate_chebyshev(
            Ciphertext<Scheme::CKKS>& input1,
            const std::function<double(double)>& function, double a, double b,
            int degree, Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Returns the levels evaluate_polynomial (with chebyshev =
         * false) or evaluate_chebyshev uses for a polynomial of the given
         * degree on [a, b].
         */
        __host__ static int polynomial_depth(int degree, bool chebyshev = false,
                                             double a = -1.0, double b = 1.0);

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns input1 + input2; see add(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        add(Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            add(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 - input2; see sub(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        sub(Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            sub(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 * input2; see multiply(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        multiply(Ciphertext<Scheme::CKKS>& input1,
                 Ciphertext<Scheme::CKKS>& input2,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            multiply(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns -input1; see negate(input1, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        negate(Ciphertext<Scheme::CKKS>& input1,
               const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            negate(input1, output, options);
            return output;
        }

        /**
         * @brief Returns the result of add_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        add_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            add_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of sub_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        sub_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            sub_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of multiply_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        multiply_plain(Ciphertext<Scheme::CKKS>& input1,
                       Plaintext<Scheme::CKKS>& input2,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            multiply_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 rotated by shift; see rotate_rows(input1,
         * output, galois_key, shift).
         */
        __host__ Ciphertext<Scheme::CKKS>
        rotate_rows(Ciphertext<Scheme::CKKS>& input1,
                    Galoiskey<Scheme::CKKS>& galois_key, int shift,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            rotate_rows(input1, output, galois_key, shift, options);
            return output;
        }

        /**
         * @brief Returns the result of add_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        add_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            add_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of sub_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        sub_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            sub_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of multiply_plain(input1, input2, output,
         * scale).
         */
        __host__ Ciphertext<Scheme::CKKS>
        multiply_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                       double scale,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            multiply_plain(input1, input2, output, scale, options);
            return output;
        }

        /**
         * @brief Returns input1 with one modulus dropped; see mod_drop(input1,
         * output).
         */
        __host__ Ciphertext<Scheme::CKKS>
        mod_drop(Ciphertext<Scheme::CKKS>& input1,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            mod_drop(input1, output, options);
            return output;
        }

        /**
         * @brief Returns input1 with one modulus dropped; see mod_drop(input1,
         * output).
         */
        __host__ Plaintext<Scheme::CKKS>
        mod_drop(Plaintext<Scheme::CKKS>& input1,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            mod_drop(input1, output, options);
            return output;
        }

        /**
         * @brief Returns the complex conjugate of input1; see conjugate(input1,
         * output, conjugate_key).
         */
        __host__ Ciphertext<Scheme::CKKS>
        conjugate(Ciphertext<Scheme::CKKS>& input1,
                  Galoiskey<Scheme::CKKS>& conjugate_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            conjugate(input1, output, conjugate_key, options);
            return output;
        }

      protected:
        __host__ void add_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                     Plaintext<Scheme::CKKS>& input2,
                                     Ciphertext<Scheme::CKKS>& output,
                                     const cudaStream_t stream);

        __host__ void add_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                             Plaintext<Scheme::CKKS>& input2,
                                             const cudaStream_t stream);

        __host__ void add_constant_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                              double input2,
                                              Ciphertext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        add_constant_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                        double input2,
                                        const cudaStream_t stream);

        __host__ void sub_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                     Plaintext<Scheme::CKKS>& input2,
                                     Ciphertext<Scheme::CKKS>& output,
                                     const cudaStream_t stream);

        __host__ void sub_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                             Plaintext<Scheme::CKKS>& input2,
                                             const cudaStream_t stream);

        __host__ void sub_constant_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                              double input2,
                                              Ciphertext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        sub_constant_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                        double input2,
                                        const cudaStream_t stream);

        __host__ void add_constant_plain_complex_ckks(
            Ciphertext<Scheme::CKKS>& input1, Complex64 c,
            Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream);

        __host__ void multiply_const_plain_complex_ckks(
            Ciphertext<Scheme::CKKS>& input1, Complex64 c,
            Ciphertext<Scheme::CKKS>& output, const cudaStream_t stream);

        __host__ void scale_up_ckks(Ciphertext<Scheme::CKKS>& input,
                                    double scale,
                                    Ciphertext<Scheme::CKKS>& output,
                                    const cudaStream_t stream);

        __host__ void mult_i_ckks(Ciphertext<Scheme::CKKS>& input1,
                                  Ciphertext<Scheme::CKKS>& output,
                                  const cudaStream_t stream);

        __host__ void div_i_ckks(Ciphertext<Scheme::CKKS>& input1,
                                 Ciphertext<Scheme::CKKS>& output,
                                 const cudaStream_t stream);

        __host__ void multiply_ckks(Ciphertext<Scheme::CKKS>& input1,
                                    Ciphertext<Scheme::CKKS>& input2,
                                    Ciphertext<Scheme::CKKS>& output,
                                    const cudaStream_t stream);

        __host__ void multiply_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                          Plaintext<Scheme::CKKS>& input2,
                                          Ciphertext<Scheme::CKKS>& output,
                                          const cudaStream_t stream);

        __host__ void
        multiply_const_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                  double input2,
                                  Ciphertext<Scheme::CKKS>& output,
                                  double scale, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void relinearize_external_product_method2_inplace_ckks(
            Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
            const cudaStream_t stream);

        ///////////////////////////////////////////////////

        // c0 and c1 of a ciphertext in coefficient form and the digits of c1
        // in P * Q_l, NTT form: the part of a Galois automorphism that does
        // not depend on the Galois element, which automorphisms of one
        // ciphertext can share.
        struct GaloisDecomposition
        {
            DeviceVector<Data64> coefficients;
            DeviceVector<Data64> digits;
        };

        __host__ GaloisDecomposition decompose_for_galois(
            Ciphertext<Scheme::CKKS>& input1, const cudaStream_t stream);

        // Applies the automorphism of galois_elt to input1, given its
        // decomposition.
        __host__ void
        apply_galois_decomposed(Ciphertext<Scheme::CKKS>& input1,
                                GaloisDecomposition& decomposition,
                                Ciphertext<Scheme::CKKS>& output,
                                Galoiskey<Scheme::CKKS>& galois_key,
                                int galois_elt, const cudaStream_t stream);

        // The ModUp of key switching: splits input, the depth_-th level
        // polynomial to be key switched in coefficient form, into the digits
        // of the context and extends each digit to the primes of P * Q_l.
        __host__ void decompose_to_Qtilde(Data64* input, Data64* output,
                                          int depth, cudaStream_t stream);

        __host__ void rotate_ckks_method_II(Ciphertext<Scheme::CKKS>& input1,
                                            Ciphertext<Scheme::CKKS>& output,
                                            Galoiskey<Scheme::CKKS>& galois_key,
                                            int shift,
                                            const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        apply_galois_ckks_method_II(Ciphertext<Scheme::CKKS>& input1,
                                    Ciphertext<Scheme::CKKS>& output,
                                    Galoiskey<Scheme::CKKS>& galois_key,
                                    int galois_elt, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void switchkey_ckks_method_II(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Switchkey<Scheme::CKKS>& switch_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void conjugate_ckks_method_II(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Galoiskey<Scheme::CKKS>& conjugate_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        rescale_inplace_ckks_leveled(Ciphertext<Scheme::CKKS>& input1,
                                     const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void mod_drop_ckks_leveled(Ciphertext<Scheme::CKKS>& input1,
                                            Ciphertext<Scheme::CKKS>& output,
                                            const cudaStream_t stream);

        __host__ void mod_drop_ckks_plaintext(Plaintext<Scheme::CKKS>& input1,
                                              Plaintext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        mod_drop_ckks_plaintext_inplace(Plaintext<Scheme::CKKS>& input1,
                                        const cudaStream_t stream);

        __host__ void
        mod_drop_ckks_leveled_inplace(Ciphertext<Scheme::CKKS>& input1,
                                      const cudaStream_t stream);

      protected:
        HEContext<Scheme::CKKS> context_;

        // The operators of HELogicOperator always work as in
        // rescale_mode::manual.
        bool manual_only_ = false;

        // True in rescale_mode::automatic (see
        // HEContextImpl<Scheme::CKKS>::set_rescale_mode), unless the operator
        // is manual_only_ or a ManualScope exists on this thread.
        bool automatic() const;

        // While it exists, the operations on this thread work as in
        // rescale_mode::manual. The operations built from other operations
        // plan their levels and scales themselves.
        class ManualScope
        {
          public:
            ManualScope();
            ~ManualScope();
            ManualScope(const ManualScope&) = delete;
            ManualScope& operator=(const ManualScope&) = delete;
        };

        // Throws StateError in rescale_mode::automatic, for the operations
        // it does not support.
        void require_manual(const char* operation) const;

        // Returns input rescaled if it needs a rescaling: input itself, or
        // storage holding the rescaled copy.
        Ciphertext<Scheme::CKKS>& rescaled(Ciphertext<Scheme::CKKS>& input,
                                           Ciphertext<Scheme::CKKS>& storage,
                                           const ExecutionOptions& options);

        // Returns input, which does not need a rescaling, brought down to
        // depth with the scale of that level: it is dropped to the level
        // above, multiplied by the integer closest to the target scale times
        // the prime of that level over its scale, and rescaled. Returns input
        // itself if it is at depth, else storage.
        Ciphertext<Scheme::CKKS>& dropped(Ciphertext<Scheme::CKKS>& input,
                                          int depth,
                                          Ciphertext<Scheme::CKKS>& storage,
                                          const ExecutionOptions& options);

        // Throws ParameterError unless input has the scale of its level, or
        // its square if input needs a rescaling.
        void check_level_scale(const Ciphertext<Scheme::CKKS>& input) const;
        void check_level_scale(const Plaintext<Scheme::CKKS>& input) const;

        // Brings two operands to the same level and to the scales of that
        // level. Both are rescaled first if they need it, unless both need
        // it at the same level and keep_rescale is true.
        void prepare(Ciphertext<Scheme::CKKS>*& input1,
                     Ciphertext<Scheme::CKKS>*& input2,
                     Ciphertext<Scheme::CKKS>& storage1,
                     Ciphertext<Scheme::CKKS>& storage2, bool keep_rescale,
                     const ExecutionOptions& options);

        // Brings a ciphertext and a plaintext to the same level and checks
        // their scales; the ciphertext is rescaled first if it needs it.
        void prepare(Ciphertext<Scheme::CKKS>*& input1,
                     Plaintext<Scheme::CKKS>*& input2,
                     Ciphertext<Scheme::CKKS>& storage1,
                     Plaintext<Scheme::CKKS>& storage2,
                     const ExecutionOptions& options);

        // The input of an operation that plans its own levels: in
        // rescale_mode::automatic, input rescaled if it needs it, with the
        // scale of its level.
        Ciphertext<Scheme::CKKS>&
        automatic_input(Ciphertext<Scheme::CKKS>& input,
                        Ciphertext<Scheme::CKKS>& storage,
                        const ExecutionOptions& options);

        // The scale of the level depth levels below input in
        // rescale_mode::automatic; 0 otherwise.
        double automatic_target(const Ciphertext<Scheme::CKKS>& input,
                                int depth) const;

        // In rescale_mode::automatic, runs operation on the prepared operands
        // as in rescale_mode::manual and returns true; returns false
        // otherwise.
        template <typename Operation>
        bool run_automatic(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2, bool keep_rescale,
                           const ExecutionOptions& options, Operation operation)
        {
            if (!automatic())
            {
                return false;
            }
            Ciphertext<Scheme::CKKS> storage1, storage2;
            Ciphertext<Scheme::CKKS>* first = &input1;
            Ciphertext<Scheme::CKKS>* second = &input2;
            prepare(first, second, storage1, storage2, keep_rescale, options);
            ManualScope manual;
            operation(*first, *second);
            return true;
        }

        template <typename Operation>
        bool run_automatic(Ciphertext<Scheme::CKKS>& input1,
                           Plaintext<Scheme::CKKS>& input2,
                           const ExecutionOptions& options, Operation operation)
        {
            if (!automatic())
            {
                return false;
            }
            Ciphertext<Scheme::CKKS> storage1;
            Plaintext<Scheme::CKKS> storage2;
            Ciphertext<Scheme::CKKS>* first = &input1;
            Plaintext<Scheme::CKKS>* second = &input2;
            prepare(first, second, storage1, storage2, options);
            ManualScope manual;
            operation(*first, *second);
            return true;
        }

        // For operations of one ciphertext, which is rescaled first if it
        // needs it.
        template <typename Operation>
        bool run_automatic(Ciphertext<Scheme::CKKS>& input1,
                           const ExecutionOptions& options, Operation operation)
        {
            if (!automatic())
            {
                return false;
            }
            Ciphertext<Scheme::CKKS> storage;
            Ciphertext<Scheme::CKKS>& first =
                rescaled(input1, storage, options);
            ManualScope manual;
            operation(first);
            return true;
        }

        // Temp(to avoid allocation time)
        DeviceVector<int> new_prime_locations_;
        DeviceVector<int> new_input_locations_;
        int* new_prime_locations;
        int* new_input_locations;

        // private:
      protected:
        __host__ Plaintext<Scheme::CKKS>
        operator_plaintext(cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Plaintext<Scheme::CKKS>
        operator_from_plaintext(Plaintext<Scheme::CKKS>& input,
                                cudaStream_t stream = cudaStreamDefault);

        __host__ Ciphertext<Scheme::CKKS>
        operator_ciphertext(double scale,
                            cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Ciphertext<Scheme::CKKS>
        operator_from_ciphertext(Ciphertext<Scheme::CKKS>& input,
                                 cudaStream_t stream = cudaStreamDefault);

        __host__ std::vector<int> rotation_index_generator(uint64_t n, int K,
                                                           int M);

        class Vandermonde
        {
            template <Scheme S> friend class HEOperator;
            template <Scheme S> friend class HEArithmeticOperator;
            template <Scheme S> friend class HELogicOperator;

          public:
            __host__ Vandermonde(const int poly_degree, const int CtoS_piece,
                                 const int StoC_piece,
                                 const double CtoS_Scaling,
                                 const double StoC_Scaling,
                                 const float CtoS_bsgs_ratio,
                                 const float StoC_bsgs_ratio);

            __host__ void generate_E_diagonals_index();

            __host__ void generate_E_inv_diagonals_index();

            __host__ void split_E();

            __host__ void split_E_inv();

            __host__ void generate_E_diagonals();

            __host__ void generate_E_inv_diagonals();

            __host__ void generate_V_n_lists(float CtoS_bsgs_ratio,
                                             float StoC_bsgs_ratio);

            __host__ void generate_pre_comp_V();

            __host__ void generate_pre_comp_V_inv();

            __host__ void generate_key_indexs();

            Vandermonde() = delete;

          private:
            int poly_degree_;
            int num_slots_;
            int log_num_slots_;

            int CtoS_piece_;
            int StoC_piece_;

            double CtoS_Scaling_;
            double StoC_Scaling_;

            std::vector<int> E_size_;
            std::vector<int> E_inv_size_;

            std::vector<int> E_index_;
            std::vector<int> E_inv_index_;

            std::vector<int> E_splitted_;
            std::vector<int> E_inv_splitted_;

            std::vector<std::vector<int>> E_splitted_index_;
            std::vector<std::vector<int>> E_inv_splitted_index_;

            std::vector<std::vector<int>> E_splitted_diag_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_diag_index_gpu_;

            std::vector<std::vector<int>> E_splitted_input_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_input_index_gpu_;

            std::vector<std::vector<int>> E_splitted_output_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_output_index_gpu_;

            std::vector<std::vector<int>> E_splitted_iteration_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_iteration_gpu_;

            std::vector<std::vector<int>> V_matrixs_index_;
            std::vector<std::vector<int>> V_inv_matrixs_index_;

            std::vector<heongpu::DeviceVector<Complex64>> V_matrixs_;
            std::vector<heongpu::DeviceVector<Complex64>> V_inv_matrixs_;

            std::vector<heongpu::DeviceVector<Complex64>> V_matrixs_rotated_;
            std::vector<heongpu::DeviceVector<Complex64>>
                V_inv_matrixs_rotated_;

            std::vector<std::vector<std::vector<int>>> diags_matrices_bsgs_;
            std::vector<std::vector<std::vector<int>>> diags_matrices_inv_bsgs_;

            std::vector<std::vector<int>> diags_matrices_bsgs_rot_n1_;
            std::vector<std::vector<int>> diags_matrices_bsgs_rot_n2_;

            std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n1_;
            std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n2_;

            std::vector<int> key_indexs_;
        };

        class Polynomial
        {
            template <Scheme S> friend class HEOperator;
            template <Scheme S> friend class HEArithmeticOperator;

          public:
            __host__ Polynomial()
                : max_deg_(0), lead_(false), type_(PolyType::CHEBYSHEV),
                  a_(0.0), b_(0.0)
            {
            }

            // type: POLY_TYPE::MONOMIAL or POLY_TYPE::CHEBYSHEV
            __host__ Polynomial(int max_deg,
                                const std::vector<Complex64>& coeffs,
                                bool lead = false,
                                PolyType type = PolyType::CHEBYSHEV,
                                double a = 0.0, double b = 0.0);

            // Get degree of polynomial (number of coefficients - 1)
            __host__ int degree() const;

            // Get depth (ceil(log2(degree)))
            __host__ int depth() const;

            // Split polynomial coefficients for BSGS recursion
            // Returns pair<poly_q, poly_r>
            __host__ std::pair<Polynomial, Polynomial>
            split_coeffs(int split) const;

            PolyType type_;
            int max_deg_;
            std::vector<Complex64> coeffs_;
            bool lead_;
            double a_, b_;
        };

        double scale_boot_;
        bool boot_context_generated_ = false;

        int CtoS_piece_;
        int StoC_piece_;

        EncodingMatrixConfig cts_config_;
        EncodingMatrixConfig stc_config_;
        EvalModConfig eval_mod_config_;

        Polynomial sine_poly_;

        std::vector<int> key_indexs_;

        std::vector<heongpu::DeviceVector<Data64>> V_matrixs_rotated_encoded_;
        std::vector<heongpu::DeviceVector<Data64>>
            V_inv_matrixs_rotated_encoded_;

        std::vector<std::vector<int>> V_matrixs_index_;
        std::vector<std::vector<int>> V_inv_matrixs_index_;

        std::vector<std::vector<std::vector<int>>> diags_matrices_bsgs_;
        std::vector<std::vector<std::vector<int>>> diags_matrices_inv_bsgs_;

        std::vector<std::vector<int>> diags_matrices_bsgs_rot_n1_;
        std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n1_;

        std::vector<std::vector<int>> diags_matrices_bsgs_rot_n2_;
        std::vector<std::vector<int>> diags_matrices_inv_bsgs_rot_n2_;

        ///////// Operator Class Encode Fuctions //////////

        int slot_count_;
        int log_slot_count_;

        // CKKS
        double two_pow_64_;
        std::shared_ptr<DeviceVector<int>> reverse_order_;
        std::shared_ptr<DeviceVector<Complex64>> special_ifft_roots_table_;

        __host__ void quick_ckks_encoder_vec_complex(Complex64* input,
                                                     Data64* output,
                                                     const double scale,
                                                     int rns_count);

        __host__ void quick_ckks_encoder_constant_complex(Complex64 input,
                                                          Data64* output,
                                                          const double scale);

        __host__ void quick_ckks_encoder_constant_double(double input,
                                                         Data64* output,
                                                         const double scale);

        __host__ void quick_ckks_encoder_constant_integer(std::int64_t input,
                                                          Data64* output,
                                                          const double scale);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_matrixs(Vandermonde& vandermonde, int start_level);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_inv_matrixs(Vandermonde& vandermonde, int start_level);

        __host__ Polynomial generate_eval_mod_poly(const EvalModConfig& config,
                                                   int max_deg);

        ///////////////////////////////////////////////////

        __host__ Ciphertext<Scheme::CKKS> multiply_matrix(
            Ciphertext<Scheme::CKKS>& cipher,
            std::vector<heongpu::DeviceVector<Data64>>& matrix,
            std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
            std::vector<std::vector<int>>& diags_matrices_bsgs_rot_n1_,
            std::vector<std::vector<int>>& diags_matrices_bsgs_rot_n2_,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ std::vector<Ciphertext<Scheme::CKKS>>
        coeff_to_slot(Ciphertext<Scheme::CKKS>& cipher,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      CKKSEncodingTransformContext& transform_context,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ std::vector<Ciphertext<Scheme::CKKS>>
        coeff_to_slot(Ciphertext<Scheme::CKKS>& cipher,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> solo_coeff_to_slot(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        slot_to_coeff(Ciphertext<Scheme::CKKS>& cipher0,
                      Ciphertext<Scheme::CKKS>& cipher1,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      CKKSEncodingTransformContext& transform_context,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        slot_to_coeff(Ciphertext<Scheme::CKKS>& cipher0,
                      Ciphertext<Scheme::CKKS>& cipher1,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> solo_slot_to_coeff(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        mod_up_from_q0(Ciphertext<Scheme::CKKS>& cipher,
                       Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
                       Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
                       const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        eval_mod(Ciphertext<Scheme::CKKS>& cipher,
                 Relinkey<Scheme::CKKS>& relin_key,
                 const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        eval_mod(Ciphertext<Scheme::CKKS>& cipher, const Polynomial& poly,
                 double shift, Relinkey<Scheme::CKKS>& relin_key,
                 const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        gen_power(std::unordered_map<int, Ciphertext<Scheme::CKKS>>& cipher,
                  int power, PolyType type, Relinkey<Scheme::CKKS>& relin_key,
                  const ExecutionOptions& options = ExecutionOptions());

        static Switchkey<Scheme::CKKS>*
        key_or_null(std::optional<Switchkey<Scheme::CKKS>>& key)
        {
            return key ? &*key : nullptr;
        }

        // Evaluates the polynomial in the given basis on input1 for
        // evaluate_polynomial and evaluate_chebyshev.
        __host__ Ciphertext<Scheme::CKKS>
        // Evaluates the polynomial with the result at target_scale; 0 for the
        // scale of input1.
        evaluate_in_basis(Ciphertext<Scheme::CKKS>& input1,
                          const std::vector<double>& coefficients,
                          PolyType type, Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options,
                          double target_scale = 0.0);

        // base_scale is the scale of the computation: intermediate results
        // with at least half of it are rescaled.
        __host__ Ciphertext<Scheme::CKKS>
        evaluate_poly(Ciphertext<Scheme::CKKS>& cipher, double target_scale,
                      double base_scale, const Polynomial& pol,
                      Relinkey<Scheme::CKKS>& relin_key,
                      const ExecutionOptions& options);

        __host__ Ciphertext<Scheme::CKKS> evaluate_poly_from_polynomial_basis(
            double target_scale, int target_level, const Polynomial& pol,
            std::unordered_map<int, Ciphertext<Scheme::CKKS>>& powered_ciphers,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> evaluate_poly_recurse(
            int target_level, double target_scale, double base_scale,
            const Polynomial& pol, int log_split,
            std::unordered_map<int, Ciphertext<Scheme::CKKS>>& powered_ciphers,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());
    };

    /**
     * @brief HEArithmeticOperator performs arithmetic operations on
     * ciphertexts.
     */
    template <>
    class HEArithmeticOperator<Scheme::CKKS> : public HEOperator<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEArithmeticOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for arithmetic operations.
         */
        HEArithmeticOperator(HEContext<Scheme::CKKS> context,
                             HEEncoder<Scheme::CKKS>& encoder);

        __host__ void generate_encoding_transform_context(
            CKKSEncodingTransformContext& transform_context,
            const CKKSEncodingTransformConfig& config =
                CKKSEncodingTransformConfig());

        /**
         * @brief Generates bootstrapping parameters.
         *
         * @param scale Scaling factor.
         * @param config Bootstrapping configuration.
         * @param boot_type Regular or slim bootstrapping.
         */
        __host__ void generate_bootstrapping_params(
            const double scale, const BootstrappingConfig& config,
            const arithmetic_bootstrapping_type& boot_type =
                arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

        __host__ std::vector<int> bootstrapping_key_indices()
        {
            if (!boot_context_generated_)
            {
                throw std::invalid_argument(
                    "Bootstrapping key indices can not be returned before "
                    "generating Bootstrapping parameters!");
            }

            return key_indexs_;
        }

        [[deprecated("Use bootstrapping_key_indices() instead.")]] __host__
            std::vector<int>
            bootstrapping_key_indexs()
        {
            return bootstrapping_key_indices();
        }

        /**
         * @brief Performs regular bootstrapping with non-sparse key support.
         *
         * Implements the optimized bootstrapping algorithm from the paper:
         * "Efficient Bootstrapping for Approximate Homomorphic Encryption with
         * Non-Sparse Keys" (https://eprint.iacr.org/2020/1203.pdf)
         *
         * @param input1 Input ciphertext; one above level 0 is brought down
         * to it.
         * @param galois_key Galois key for rotation operations.
         * @param relin_key Relinearization key for EvalMod polynomial
         * evaluation.
         * @param swk_dense_to_sparse Optional switch key from dense to sparse
         * representation.
         * @param swk_sparse_to_dense Optional switch key from sparse to dense
         * representation.
         * @param options Execution options (stream, storage, etc.).
         * @return Ciphertext Bootstrapped ciphertext with refreshed modulus
         * chain.
         *
         * @throws std::invalid_argument if bootstrapping context not
         * initialized.
         */
        __host__ Ciphertext<Scheme::CKKS> regular_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Performs slim bootstrapping (supports only real numbers).
         *
         * @param input1 Input ciphertext at the SlotToCoeff start level or
         * above; one above it is brought down to it. A lower one is reported
         * with LevelMismatchError.
         * @param galois_key Galois key.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped ciphertext.
         */
        __host__ Ciphertext<Scheme::CKKS> slim_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates the bootstrapping parameters of preset, which must
         * be a regular or slim preset.
         */
        __host__ void
        generate_bootstrapping_params(const CKKSBootstrappingPreset& preset);

        /**
         * @brief Performs regular bootstrapping with the keys of
         * HEKeyGenerator::bootstrapping_keys; see regular_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS> regular_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, CKKSBootstrappingKeys& keys,
            const ExecutionOptions& options = ExecutionOptions())
        {
            return regular_bootstrapping(
                input1, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Performs slim bootstrapping with the keys of
         * HEKeyGenerator::bootstrapping_keys; see slim_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        slim_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                           CKKSBootstrappingKeys& keys,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return slim_bootstrapping(input1, keys.galois_key, keys.relin_key,
                                      key_or_null(keys.dense_to_sparse),
                                      key_or_null(keys.sparse_to_dense),
                                      options);
        }

        using HEOperator<Scheme::CKKS>::coeff_to_slot;
        using HEOperator<Scheme::CKKS>::slot_to_coeff;
    };

    /**
     * @brief HELogicOperator performs homomorphic logical operations on
     * ciphertexts.
     */
    template <>
    class HELogicOperator<Scheme::CKKS> : private HEOperator<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HELogicOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for homomorphic operations.
         */
        HELogicOperator(HEContext<Scheme::CKKS> context,
                        HEEncoder<Scheme::CKKS>& encoder, double scale);

        /**
         * @brief Performs logical NOT on ciphertext.
         *
         * @param input1 First input ciphertext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOT(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        { one_minus_cipher(input1_, output_, options_inner); },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOT on ciphertext.
         *
         * @param input1 Ciphertext updated with result.
         * @param options Execution options.
         */
        __host__ void
        NOT_inplace(Ciphertext<Scheme::CKKS>& input1,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                { one_minus_cipher_inplace(input1_, options_inner); },
                options, true);
        }

        /**
         * @brief Performs logical AND on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void AND(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

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
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
                                                        options_inner);
                                    rescale_inplace(output_, options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical AND on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        AND_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            AND(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical AND on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void AND(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_, output_,
                                                   options_inner);
                                    rescale_inplace(output_, options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical AND on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        AND_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            AND(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical OR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void OR(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         Ciphertext<Scheme::CKKS>& output,
                         Relinkey<Scheme::CKKS>& relin_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical OR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        OR_inplace(Ciphertext<Scheme::CKKS>& input1,
                   Ciphertext<Scheme::CKKS>& input2,
                   Relinkey<Scheme::CKKS>& relin_key,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            OR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical OR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void OR(Ciphertext<Scheme::CKKS>& input1,
                         Plaintext<Scheme::CKKS>& input2,
                         Ciphertext<Scheme::CKKS>& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical OR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        OR_inplace(Ciphertext<Scheme::CKKS>& input1,
                   Plaintext<Scheme::CKKS>& input2,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            OR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical XOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void XOR(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        XOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            XOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical XOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void XOR(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XOR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        XOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            XOR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical NAND on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void NAND(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           Relinkey<Scheme::CKKS>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

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
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
                                                        options_inner);
                                    rescale_inplace(output_, options_inner);
                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NAND on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& input2,
                     Relinkey<Scheme::CKKS>& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            NAND(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical NAND on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NAND(Ciphertext<Scheme::CKKS>& input1,
                           Plaintext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_, output_,
                                                   options_inner);
                                    rescale_inplace(output_, options_inner);
                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NAND on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Plaintext<Scheme::CKKS>& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            NAND(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical NOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void NOR(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        NOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            NOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical NOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOR(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        NOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            NOR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical XNOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void XNOR(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           Relinkey<Scheme::CKKS>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XNOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        XNOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& input2,
                     Relinkey<Scheme::CKKS>& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            XNOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical XNOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void XNOR(Ciphertext<Scheme::CKKS>& input1,
                           Plaintext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    mod_drop_inplace(inner_input2);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XNOR on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        XNOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Plaintext<Scheme::CKKS>& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input2.validate_context(context_);

            input1.validate_context(context_);

            XNOR(input1, input2, input1, options);
        }

        /**
         * @brief Generates bootstrapping parameters.
         *
         * @param scale Scaling factor.
         * @param config Bootstrapping configuration.
         * @param boot_type Bit or gate bootstrapping.
         */
        __host__ void generate_bootstrapping_params(
            const double scale, const BootstrappingConfig& config,
            const logic_bootstrapping_type& boot_type);

        /**
         * @brief Retrieves galois key indexes required for bootstrapping.
         *
         * @return std::vector<int> Bootstrapping key indexes.
         */
        __host__ std::vector<int> bootstrapping_key_indices()
        {
            if (!boot_context_generated_)
            {
                throw std::invalid_argument(
                    "Bootstrapping key indices can not be returned before "
                    "generating Bootstrapping parameters!");
            }

            return key_indexs_;
        }

        [[deprecated("Use bootstrapping_key_indices() instead.")]] __host__
            std::vector<int>
            bootstrapping_key_indexs()
        {
            return bootstrapping_key_indices();
        }

        /**
         * @brief Performs bit-level bootstrapping on a ciphertext.(For more
         * detail please check README.md)
         *
         * @param input1 Input ciphertext at the SlotToCoeff start level or
         * above; one above it is brought down to it. A lower one is reported
         * with LevelMismatchError.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped ciphertext.
         */
        __host__ Ciphertext<Scheme::CKKS> bit_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Bootstrapped logical AND gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS> AND_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::AND, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Bootstrapped logical OR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS>
        OR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key,
                         Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
                         Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::OR, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Bootstrapped logical XOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS> XOR_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::XOR, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Bootstrapped logical NAND gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS> NAND_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::NAND, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Bootstrapped logical NOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS> NOR_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::NOR, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Bootstrapped logical XNOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param swk_dense_to_sparse Optional dense to sparse switch key.
         * @param swk_sparse_to_dense Optional sparse to dense switch key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext<Scheme::CKKS> XNOR_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse = nullptr,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense = nullptr,
            const ExecutionOptions& options = ExecutionOptions())
        {
            galois_key.validate_context(context_);
            relin_key.validate_context(context_);

            input1.validate_context(context_);
            input2.validate_context(context_);

            return gate_bootstrapping(
                logic_gate::XNOR, input1, input2, galois_key, relin_key,
                swk_dense_to_sparse, swk_sparse_to_dense, options);
        }

        /**
         * @brief Generates the bootstrapping parameters of preset, which must
         * be a bit or gate preset.
         */
        __host__ void
        generate_bootstrapping_params(const CKKSBootstrappingPreset& preset);

        /**
         * @brief Performs bit-level bootstrapping with the keys of
         * HEKeyGenerator::bootstrapping_keys; see bit_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        bit_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          CKKSBootstrappingKeys& keys,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return bit_bootstrapping(input1, keys.galois_key, keys.relin_key,
                                     key_or_null(keys.dense_to_sparse),
                                     key_or_null(keys.sparse_to_dense),
                                     options);
        }

        /**
         * @brief Bootstrapped logical AND gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see AND_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        AND_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          CKKSBootstrappingKeys& keys,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return AND_bootstrapping(
                input1, input2, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Bootstrapped logical OR gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see OR_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        OR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         CKKSBootstrappingKeys& keys,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            return OR_bootstrapping(input1, input2, keys.galois_key,
                                    keys.relin_key,
                                    key_or_null(keys.dense_to_sparse),
                                    key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Bootstrapped logical XOR gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see XOR_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        XOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          CKKSBootstrappingKeys& keys,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return XOR_bootstrapping(
                input1, input2, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Bootstrapped logical NAND gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see NAND_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        NAND_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           CKKSBootstrappingKeys& keys,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return NAND_bootstrapping(
                input1, input2, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Bootstrapped logical NOR gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see NOR_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        NOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          CKKSBootstrappingKeys& keys,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return NOR_bootstrapping(
                input1, input2, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        /**
         * @brief Bootstrapped logical XNOR gate with the keys of
         * HEKeyGenerator::bootstrapping_keys; see XNOR_bootstrapping.
         */
        __host__ Ciphertext<Scheme::CKKS>
        XNOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           CKKSBootstrappingKeys& keys,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return XNOR_bootstrapping(
                input1, input2, keys.galois_key, keys.relin_key,
                key_or_null(keys.dense_to_sparse),
                key_or_null(keys.sparse_to_dense), options);
        }

        using HEOperator<Scheme::CKKS>::apply_galois;
        using HEOperator<Scheme::CKKS>::apply_galois_inplace;
        using HEOperator<Scheme::CKKS>::keyswitch;
        using HEOperator<Scheme::CKKS>::mod_drop;
        using HEOperator<Scheme::CKKS>::mod_drop_inplace;
        using HEOperator<Scheme::CKKS>::rotate_rows;
        using HEOperator<Scheme::CKKS>::rotate_rows_inplace;

      private:
        enum class logic_gate
        {
            AND,
            OR,
            XOR,
            NAND,
            NOR,
            XNOR
        };

        __host__ Ciphertext<Scheme::CKKS> gate_bootstrapping(
            logic_gate gate_type, Ciphertext<Scheme::CKKS>& input1,
            Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> logic_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1, int function_index,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Switchkey<Scheme::CKKS>* swk_dense_to_sparse,
            Switchkey<Scheme::CKKS>* swk_sparse_to_dense,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        one_minus_cipher(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& output,
                         const ExecutionOptions& options = ExecutionOptions());

        __host__ void one_minus_cipher_inplace(
            Ciphertext<Scheme::CKKS>& input1,
            const ExecutionOptions& options = ExecutionOptions());

        // Encoded One
        DeviceVector<Data64> encoded_constant_one_;

        // Bit/gate bootstrapping: one cosine polynomial, input shift and
        // output constant per function (bit, or AND..XNOR in logic_gate order)
        std::vector<Polynomial> logic_polys_;
        std::vector<double> logic_shifts_;
        std::vector<double> logic_constants_;
    };

} // namespace heongpu

#endif // HEONGPU_CKKS_OPERATOR_H
