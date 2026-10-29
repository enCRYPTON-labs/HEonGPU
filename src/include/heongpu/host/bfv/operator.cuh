// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_OPERATOR_H
#define HEONGPU_BFV_OPERATOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpufft/fft.cuh"
#include <heongpu/kernel/addition.cuh>
#include <heongpu/kernel/multiplication.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/bootstrapping.cuh>

#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/encoder.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/ciphertext.cuh>
#include <heongpu/host/bfv/evaluationkey.cuh>

namespace heongpu
{

    /**
     * @brief HEOperator is responsible for performing homomorphic operations on
     * encrypted data, such as addition, subtraction, multiplication, and other
     * functions.
     *
     * The HEOperator class is initialized with encryption parameters and
     * provides various functions for performing operations on ciphertexts,
     * including BFV scheme. It supports both in-place and
     * out-of-place operations, as well as asynchronous processing using CUDA
     * streams.
     */
    template <> class HEOperator<Scheme::BFV>
    {
      protected:
        /**
         * @brief Construct a new HEOperator object with the given parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters for the operator.
         */
        __host__ HEOperator(HEContext<Scheme::BFV> context,
                            HEEncoder<Scheme::BFV>& encoder);

      public:
        /**
         * @brief Adds two ciphertexts and stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add(const Ciphertext<Scheme::BFV>& input1,
                          const Ciphertext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
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
        add_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Ciphertext<Scheme::BFV>& input2,
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
        __host__ void sub(const Ciphertext<Scheme::BFV>& input1,
                          const Ciphertext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Subtracts the second ciphertext from the first, modifying the
         * first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         */
        __host__ void
        sub_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Ciphertext<Scheme::BFV>& input2,
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
        negate(const Ciphertext<Scheme::BFV>& input1,
               Ciphertext<Scheme::BFV>& output,
               const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Negates a ciphertext in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         */
        __host__ void
        negate_inplace(Ciphertext<Scheme::BFV>& input1,
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
        add_plain(const Ciphertext<Scheme::BFV>& input1,
                  const Plaintext<Scheme::BFV>& input2,
                  Ciphertext<Scheme::BFV>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    if (input1_.in_ntt_domain_ ||
                                        input2_.in_ntt_domain_)
                                    {
                                        throw std::logic_error(
                                            "BFV ciphertext or "
                                            "plaintext "
                                            "should be not in NTT "
                                            "domain");
                                    }
                                    add_plain_bfv(input1_, input2_, output_,
                                                  options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
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
        add_plain_inplace(Ciphertext<Scheme::BFV>& input1,
                          const Plaintext<Scheme::BFV>& input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                            {
                                throw std::logic_error(
                                    "BFV ciphertext or plaintext "
                                    "should be not in NTT domain");
                            }
                            add_plain_bfv_inplace(input1_, input2_,
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
        sub_plain(const Ciphertext<Scheme::BFV>& input1,
                  const Plaintext<Scheme::BFV>& input2,
                  Ciphertext<Scheme::BFV>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    if (input1_.in_ntt_domain_ ||
                                        input2_.in_ntt_domain_)
                                    {
                                        throw std::logic_error(
                                            "BFV ciphertext or "
                                            "plaintext "
                                            "should be not in NTT "
                                            "domain");
                                    }
                                    sub_plain_bfv(input1_, input2_, output_,
                                                  options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
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
        sub_plain_inplace(Ciphertext<Scheme::BFV>& input1,
                          const Plaintext<Scheme::BFV>& input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                            {
                                throw std::logic_error(
                                    "BFV ciphertext or plaintext "
                                    "should be not in NTT domain");
                            }
                            sub_plain_bfv_inplace(input1_, input2_,
                                                  options.stream_);
                        },
                        options, false);
                },
                options, true);
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
        multiply(const Ciphertext<Scheme::BFV>& input1,
                 const Ciphertext<Scheme::BFV>& input2,
                 Ciphertext<Scheme::BFV>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_ ||
                input2.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the "
                    "non-linear part! Please use relinearization operation!");
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
                                    multiply_bfv(input1_, input2_, output_,
                                                 options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 3;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
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
        multiply_inplace(Ciphertext<Scheme::BFV>& input1,
                         const Ciphertext<Scheme::BFV>& input2,
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
        multiply_plain(const Ciphertext<Scheme::BFV>& input1,
                       const Plaintext<Scheme::BFV>& input2,
                       Ciphertext<Scheme::BFV>& output,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            if (input1.memory_size() < (2 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    if (input1_.in_ntt_domain_ !=
                                        input2_.in_ntt_domain_)
                                    {
                                        throw std::logic_error(
                                            "BFV ciphertext or "
                                            "plaintext "
                                            "should be not in same "
                                            "domain");
                                    }

                                    if (input2_.size() < context_->n)
                                    {
                                        throw ParameterError(error_message(
                                            "Invalid Plaintext size! Expected "
                                            "at least ",
                                            context_->n, " coefficients, got ",
                                            input2_.size(), "."));
                                    }

                                    multiply_plain_bfv(input1_, input2_,
                                                       output_,
                                                       options.stream_);

                                    output_.scheme_ = context_->scheme_;
                                    output_.context_id_ = context_->context_id_;
                                    output_.ring_size_ = context_->n;
                                    output_.coeff_modulus_count_ =
                                        context_->Q_size;
                                    output_.cipher_size_ = 2;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
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
            Ciphertext<Scheme::BFV>& input1,
            const Plaintext<Scheme::BFV>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            multiply_plain(input1, input2, input1, options);
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         */
        __host__ void relinearize_inplace(
            Ciphertext<Scheme::BFV>& input1,
            const Relinkey<Scheme::BFV>& relin_key,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            relin_key.validate_context(context_);

            if ((!input1.relinearization_required_))
            {
                throw std::invalid_argument(
                    "Ciphertexts can not use relinearization, since no "
                    "non-linear part!");
            }

            if (input1.memory_size() < (3 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (3 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    if (input1_.in_ntt_domain_ != false)
                    {
                        throw std::invalid_argument(
                            "Ciphertext should be in intt domain");
                    }

                    relinearize_external_product_inplace(input1_, relin_key,
                                                         options.stream_);

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
        rotate_rows(const Ciphertext<Scheme::BFV>& input1,
                    Ciphertext<Scheme::BFV>& output,
                    const Galoiskey<Scheme::BFV>& galois_key, int shift,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            galois_key.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw StateError("Ciphertext can not be rotated! "
                                 "Relinearize it first.");
            }

            if (input1.memory_size() < (2 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
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
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::BFV>& output_)
                        {
                            if (input1_.in_ntt_domain_ != false)
                            {
                                throw std::invalid_argument(
                                    "Ciphertext should be in intt "
                                    "domain");
                            }

                            rotate_bfv(input1_, output_, galois_key, shift,
                                       options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
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
            Ciphertext<Scheme::BFV>& input1,
            const Galoiskey<Scheme::BFV>& galois_key, int shift,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            galois_key.validate_context(context_);

            if (shift == 0)
            {
                return;
            }

            rotate_rows(input1, input1, galois_key, shift, options);
        }

        /**
         * @brief Rotates the columns of a ciphertext and stores the result in
         * the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         */
        __host__ void
        rotate_columns(const Ciphertext<Scheme::BFV>& input1,
                       Ciphertext<Scheme::BFV>& output,
                       const Galoiskey<Scheme::BFV>& galois_key,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            galois_key.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw StateError("Ciphertext can not be rotated! "
                                 "Relinearize it first.");
            }

            if (input1.memory_size() < (2 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::BFV>& output_)
                        {
                            rotate_columns_bfv(input1_, output_, galois_key,
                                               options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
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
        apply_galois(const Ciphertext<Scheme::BFV>& input1,
                     Ciphertext<Scheme::BFV>& output,
                     const Galoiskey<Scheme::BFV>& galois_key, int galois_elt,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            galois_key.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw StateError("Ciphertext can not be rotated! "
                                 "Relinearize it first.");
            }

            if (input1.memory_size() < (2 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::BFV>& output_)
                        {
                            if (input1_.in_ntt_domain_ != false)
                            {
                                throw std::invalid_argument(
                                    "Ciphertext should be in intt "
                                    "domain");
                            }

                            apply_galois_bfv(input1_, output_, galois_key,
                                             galois_elt, options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
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
            Ciphertext<Scheme::BFV>& input1,
            const Galoiskey<Scheme::BFV>& galois_key, int galois_elt,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            galois_key.validate_context(context_);

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
        keyswitch(const Ciphertext<Scheme::BFV>& input1,
                  Ciphertext<Scheme::BFV>& output,
                  const Switchkey<Scheme::BFV>& switch_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            switch_key.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw StateError("Ciphertext can not be rotated! "
                                 "Relinearize it first.");
            }

            if (input1.memory_size() < (2 * context_->n * context_->Q_size))
            {
                throw ParameterError(error_message(
                    "Invalid Ciphertexts size! Expected at least ",
                    (2 * context_->n * context_->Q_size), " coefficients, got ",
                    input1.memory_size(), "."));
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::BFV>& output_)
                        {
                            if (input1_.in_ntt_domain_ != false)
                            {
                                throw std::invalid_argument(
                                    "Ciphertext should be in intt "
                                    "domain");
                            }

                            switchkey_bfv(input1_, output_, switch_key,
                                          options.stream_);

                            output_.scheme_ = context_->scheme_;
                            output_.context_id_ = context_->context_id_;
                            output_.ring_size_ = context_->n;
                            output_.coeff_modulus_count_ = context_->Q_size;
                            output_.cipher_size_ = 2;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        __host__ void multiply_power_of_X(
            const Ciphertext<Scheme::BFV>& input1,
            Ciphertext<Scheme::BFV>& output, int index,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (index != 0)
            {
                if (input1.in_ntt_domain_ != false)
                {
                    throw std::invalid_argument(
                        "Ciphertext should be in intt domain");
                }

                if (input1.memory_size() < (2 * context_->n * context_->Q_size))
                {
                    throw ParameterError(error_message(
                        "Invalid Ciphertexts size! Expected at least ",
                        (2 * context_->n * context_->Q_size),
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
                                negacyclic_shift_poly_coeffmod(
                                    input1_, output_, index, options.stream_);

                                output.scheme_ = context_->scheme_;
                                output.context_id_ = context_->context_id_;
                                output.ring_size_ = context_->n;
                                output.coeff_modulus_count_ = context_->Q_size;
                                output.cipher_size_ = 2;
                                output.in_ntt_domain_ = input1.in_ntt_domain_;
                                output.relinearization_required_ =
                                    input1.relinearization_required_;
                                output_.ciphertext_generated_ = true;
                            },
                            options);
                    },
                    options, (&input1 == &output));
            }
        }

        /**
         * @brief Transforms a plaintext to the NTT domain and stores the result
         * in the output.
         *
         * @param input1 Input plaintext to be transformed.
         * @param output Plaintext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_to_ntt(const Plaintext<Scheme::BFV>& input1,
                         Plaintext<Scheme::BFV>& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (!input1.in_ntt_domain_)
            {
                if (input1.size() < context_->n)
                {
                    throw ParameterError(error_message(
                        "Invalid Ciphertexts size! Expected at least ",
                        context_->n, " coefficients, got ", input1.size(),
                        "."));
                }

                input_storage_manager(
                    input1,
                    [&](Plaintext<Scheme::BFV>& input1_)
                    {
                        output_storage_manager(
                            output,
                            [&](Plaintext<Scheme::BFV>& output_)
                            {
                                transform_to_ntt_bfv_plain(input1_, output_,
                                                           options.stream_);

                                output.scheme_ = input1.scheme_;
                                output.context_id_ = input1.context_id_;
                                output.plain_size_ =
                                    (context_->n * context_->Q_size);
                                output.in_ntt_domain_ = true;
                                output_.plaintext_generated_ = true;
                            },
                            options);
                    },
                    options, (&input1 == &output));
            }
        }

        /**
         * @brief Transforms a plaintext to the NTT domain in-place, modifying
         * the input plaintext.
         *
         * @param input1 Plaintext to be transformed.
         */
        __host__ void transform_to_ntt_inplace(
            Plaintext<Scheme::BFV>& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            transform_to_ntt(input1, input1, options);
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_to_ntt(const Ciphertext<Scheme::BFV>& input1,
                         Ciphertext<Scheme::BFV>& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed to NTT!");
            }

            if (!input1.in_ntt_domain_)
            {
                if (input1.memory_size() < (2 * context_->n * context_->Q_size))
                {
                    throw ParameterError(error_message(
                        "Invalid Ciphertexts size! Expected at least ",
                        (2 * context_->n * context_->Q_size),
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
                                transform_to_ntt_bfv_cipher(input1_, output_,
                                                            options.stream_);

                                output.scheme_ = context_->scheme_;
                                output.context_id_ = context_->context_id_;
                                output.ring_size_ = context_->n;
                                output.coeff_modulus_count_ = context_->Q_size;
                                output.cipher_size_ = 2;
                                output.in_ntt_domain_ = true;
                                output.relinearization_required_ =
                                    input1.relinearization_required_;
                                output_.ciphertext_generated_ = true;
                            },
                            options);
                    },
                    options, (&input1 == &output));
            }
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed.
         */
        __host__ void transform_to_ntt_inplace(
            Ciphertext<Scheme::BFV>& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            transform_to_ntt(input1, input1, options);
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed from the NTT domain.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_from_ntt(const Ciphertext<Scheme::BFV>& input1,
                           Ciphertext<Scheme::BFV>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed from NTT!");
            }

            if (input1.in_ntt_domain_)
            {
                if (input1.memory_size() < (2 * context_->n * context_->Q_size))
                {
                    throw ParameterError(error_message(
                        "Invalid Ciphertexts size! Expected at least ",
                        (2 * context_->n * context_->Q_size),
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
                                transform_from_ntt_bfv_cipher(input1_, output_,
                                                              options.stream_);

                                output.scheme_ = context_->scheme_;
                                output.context_id_ = context_->context_id_;
                                output.ring_size_ = context_->n;
                                output.coeff_modulus_count_ = context_->Q_size;
                                output.cipher_size_ = 2;
                                output.in_ntt_domain_ = false;
                                output.relinearization_required_ =
                                    input1.relinearization_required_;
                                output_.ciphertext_generated_ = true;
                            },
                            options);
                    },
                    options, (&input1 == &output));
            }
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed from the NTT domain.
         */
        __host__ void transform_from_ntt_inplace(
            Ciphertext<Scheme::BFV>& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            transform_from_ntt(input1, input1, options);
        }

        HEOperator() = default;
        HEOperator(const HEOperator& copy) = default;
        HEOperator(HEOperator&& source) = default;
        HEOperator& operator=(const HEOperator& assign) = default;
        HEOperator& operator=(HEOperator&& assign) = default;

        /**
         * @brief Returns a ciphertext whose slots all hold the sum of the slots
         * of input1, over both rows. Uses log2(row size) rotations by the
         * steps 1, 2, 4, ..., row size / 2 and one column rotation, so
         * galois_key needs a key for each of these steps, as
         * Galoiskey(context, log2(row size) - 1) has.
         */
        __host__ Ciphertext<Scheme::BFV>
        sum_slots(const Ciphertext<Scheme::BFV>& input1,
                  const Galoiskey<Scheme::BFV>& galois_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> result = input1;
            for (int step = 1; step < (context_->n >> 1); step <<= 1)
            {
                Ciphertext<Scheme::BFV> rotated(context_, options);
                rotate_rows(result, rotated, galois_key, step, options);
                add_inplace(result, rotated, options);
            }
            Ciphertext<Scheme::BFV> swapped(context_, options);
            rotate_columns(result, swapped, galois_key, options);
            add_inplace(result, swapped, options);
            return result;
        }

        /**
         * @brief Returns a ciphertext whose slots all hold the inner product
         * of the slots of input1 and input2. Multiplies, relinearizes and then
         * calls sum_slots.
         */
        __host__ Ciphertext<Scheme::BFV>
        inner_product(const Ciphertext<Scheme::BFV>& input1,
                      const Ciphertext<Scheme::BFV>& input2,
                      const Relinkey<Scheme::BFV>& relin_key,
                      const Galoiskey<Scheme::BFV>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> product(context_, options);
            multiply(input1, input2, product, options);
            relinearize_inplace(product, relin_key, options);
            return sum_slots(product, galois_key, options);
        }

        /**
         * @brief Returns a ciphertext whose slots all hold the inner product
         * of the slots of input1 and input2. Multiplies and then calls
         * sum_slots.
         */
        __host__ Ciphertext<Scheme::BFV>
        inner_product(const Ciphertext<Scheme::BFV>& input1,
                      const Plaintext<Scheme::BFV>& input2,
                      const Galoiskey<Scheme::BFV>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> product(context_, options);
            multiply_plain(input1, input2, product, options);
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
        __host__ std::vector<Ciphertext<Scheme::BFV>> rotate_rows_hoisted(
            const Ciphertext<Scheme::BFV>& input1,
            const Galoiskey<Scheme::BFV>& galois_key,
            const std::vector<int>& shifts,
            const ExecutionOptions& options = ExecutionOptions());

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns input1 + input2; see add(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        add(const Ciphertext<Scheme::BFV>& input1,
            const Ciphertext<Scheme::BFV>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            add(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 - input2; see sub(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        sub(const Ciphertext<Scheme::BFV>& input1,
            const Ciphertext<Scheme::BFV>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            sub(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 * input2; see multiply(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        multiply(const Ciphertext<Scheme::BFV>& input1,
                 const Ciphertext<Scheme::BFV>& input2,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            multiply(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns -input1; see negate(input1, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        negate(const Ciphertext<Scheme::BFV>& input1,
               const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            negate(input1, output, options);
            return output;
        }

        /**
         * @brief Returns the result of add_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        add_plain(const Ciphertext<Scheme::BFV>& input1,
                  const Plaintext<Scheme::BFV>& input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            add_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of sub_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        sub_plain(const Ciphertext<Scheme::BFV>& input1,
                  const Plaintext<Scheme::BFV>& input2,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            sub_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns the result of multiply_plain(input1, input2, output).
         */
        __host__ Ciphertext<Scheme::BFV>
        multiply_plain(const Ciphertext<Scheme::BFV>& input1,
                       const Plaintext<Scheme::BFV>& input2,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            multiply_plain(input1, input2, output, options);
            return output;
        }

        /**
         * @brief Returns input1 rotated by shift; see rotate_rows(input1,
         * output, galois_key, shift).
         */
        __host__ Ciphertext<Scheme::BFV>
        rotate_rows(const Ciphertext<Scheme::BFV>& input1,
                    const Galoiskey<Scheme::BFV>& galois_key, int shift,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            rotate_rows(input1, output, galois_key, shift, options);
            return output;
        }

        /**
         * @brief Returns input1 with its rows swapped; see
         * rotate_columns(input1, output, galois_key).
         */
        __host__ Ciphertext<Scheme::BFV>
        rotate_columns(const Ciphertext<Scheme::BFV>& input1,
                       const Galoiskey<Scheme::BFV>& galois_key,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            rotate_columns(input1, output, galois_key, options);
            return output;
        }

        // private:
      protected:
        __host__ void add_plain_bfv(Ciphertext<Scheme::BFV>& input1,
                                    Plaintext<Scheme::BFV>& input2,
                                    Ciphertext<Scheme::BFV>& output,
                                    const cudaStream_t stream);

        __host__ void add_plain_bfv_inplace(Ciphertext<Scheme::BFV>& input1,
                                            Plaintext<Scheme::BFV>& input2,
                                            const cudaStream_t stream);

        __host__ void sub_plain_bfv(Ciphertext<Scheme::BFV>& input1,
                                    Plaintext<Scheme::BFV>& input2,
                                    Ciphertext<Scheme::BFV>& output,
                                    const cudaStream_t stream);

        __host__ void sub_plain_bfv_inplace(Ciphertext<Scheme::BFV>& input1,
                                            Plaintext<Scheme::BFV>& input2,
                                            const cudaStream_t stream);

        __host__ void multiply_bfv(Ciphertext<Scheme::BFV>& input1,
                                   Ciphertext<Scheme::BFV>& input2,
                                   Ciphertext<Scheme::BFV>& output,
                                   const cudaStream_t stream);

        __host__ void multiply_plain_bfv(Ciphertext<Scheme::BFV>& input1,
                                         Plaintext<Scheme::BFV>& input2,
                                         Ciphertext<Scheme::BFV>& output,
                                         const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void relinearize_external_product_inplace(
            Ciphertext<Scheme::BFV>& input1,
            const Relinkey<Scheme::BFV>& relin_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        // The digits of c1 of a ciphertext in P * Q, NTT form: the part of a

        // Galois automorphism that does not depend on the Galois element, which

        // automorphisms of one ciphertext can share.

        struct GaloisDecomposition

        {
            DeviceVector<Data64> digits;
        };

        __host__ GaloisDecomposition

        decompose_for_galois(Ciphertext<Scheme::BFV>& input1,

                             const cudaStream_t stream);

        // Applies the automorphism of galois_elt to input1, given its

        // decomposition.

        __host__ void apply_galois_decomposed(

            Ciphertext<Scheme::BFV>& input1, GaloisDecomposition& decomposition,

            Ciphertext<Scheme::BFV>& output,
            const Galoiskey<Scheme::BFV>& galois_key,

            int galois_elt, const cudaStream_t stream);

        // The ModUp of key switching: splits input, a polynomial in coefficient

        // form, into the digits of the context and extends each digit to the

        // primes of P * Q.

        __host__ void decompose_to_Qtilde(Data64* input, Data64* output,

                                          cudaStream_t stream);

        __host__ void rotate_bfv(Ciphertext<Scheme::BFV>& input1,
                                 Ciphertext<Scheme::BFV>& output,
                                 const Galoiskey<Scheme::BFV>& galois_key,
                                 int shift, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void apply_galois_bfv(Ciphertext<Scheme::BFV>& input1,
                                       Ciphertext<Scheme::BFV>& output,
                                       const Galoiskey<Scheme::BFV>& galois_key,
                                       int galois_elt,
                                       const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        rotate_columns_bfv(Ciphertext<Scheme::BFV>& input1,
                           Ciphertext<Scheme::BFV>& output,
                           const Galoiskey<Scheme::BFV>& galois_key,
                           const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void switchkey_bfv(Ciphertext<Scheme::BFV>& input1,
                                    Ciphertext<Scheme::BFV>& output,
                                    const Switchkey<Scheme::BFV>& switch_key,
                                    const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        negacyclic_shift_poly_coeffmod(Ciphertext<Scheme::BFV>& input1,
                                       Ciphertext<Scheme::BFV>& output,
                                       int index, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void transform_to_ntt_bfv_plain(Plaintext<Scheme::BFV>& input1,
                                                 Plaintext<Scheme::BFV>& output,
                                                 const cudaStream_t stream);

        __host__ void
        transform_to_ntt_bfv_cipher(Ciphertext<Scheme::BFV>& input1,
                                    Ciphertext<Scheme::BFV>& output,
                                    const cudaStream_t stream);

        __host__ void
        transform_from_ntt_bfv_cipher(Ciphertext<Scheme::BFV>& input1,
                                      Ciphertext<Scheme::BFV>& output,
                                      const cudaStream_t stream);

        // private:
      protected:
        HEContext<Scheme::BFV> context_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;

        // Temp(to avoid allocation time)
        DeviceVector<int> new_prime_locations_;
        DeviceVector<int> new_input_locations_;
        int* new_prime_locations = nullptr;
        int* new_input_locations = nullptr;

      protected:
        // Just for copy parameters, not memory!
        __host__ Ciphertext<Scheme::BFV>
        operator_from_ciphertext(const Ciphertext<Scheme::BFV>& input,
                                 cudaStream_t stream = cudaStreamDefault);
    };

    /**
     * @brief HEArithmeticOperator performs arithmetic operations on
     * ciphertexts.
     */
    template <>
    class HEArithmeticOperator<Scheme::BFV> : public HEOperator<Scheme::BFV>
    {
      public:
        /**
         * @brief Constructs a new HEArithmeticOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for arithmetic operations.
         */
        HEArithmeticOperator(HEContext<Scheme::BFV> context,
                             HEEncoder<Scheme::BFV>& encoder);
    };

    /**
     * @brief HELogicOperator performs homomorphic logical operations on
     * ciphertexts.
     */
    template <>
    class HELogicOperator<Scheme::BFV> : private HEOperator<Scheme::BFV>
    {
      public:
        /**
         * @brief Constructs a new HELogicOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for homomorphic operations.
         */
        HELogicOperator(HEContext<Scheme::BFV> context,
                        HEEncoder<Scheme::BFV>& encoder);

        /**
         * @brief Performs logical NOT on ciphertext.
         *
         * @param input1 First input ciphertext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOT(const Ciphertext<Scheme::BFV>& input1,
                          Ciphertext<Scheme::BFV>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::BFV>& output_)
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
        NOT_inplace(Ciphertext<Scheme::BFV>& input1,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
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
        __host__ void AND(const Ciphertext<Scheme::BFV>& input1,
                          const Ciphertext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const Relinkey<Scheme::BFV>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

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
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
                                                        options_inner);
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
        AND_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Ciphertext<Scheme::BFV>& input2,
                    const Relinkey<Scheme::BFV>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void AND(const Ciphertext<Scheme::BFV>& input1,
                          const Plaintext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_) {
                                    multiply_plain(input1_, input2_, output_,
                                                   options_inner);
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
        AND_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Plaintext<Scheme::BFV>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

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
        __host__ void OR(const Ciphertext<Scheme::BFV>& input1,
                         const Ciphertext<Scheme::BFV>& input2,
                         Ciphertext<Scheme::BFV>& output,
                         const Relinkey<Scheme::BFV>& relin_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
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
        OR_inplace(Ciphertext<Scheme::BFV>& input1,
                   const Ciphertext<Scheme::BFV>& input2,
                   const Relinkey<Scheme::BFV>& relin_key,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void OR(const Ciphertext<Scheme::BFV>& input1,
                         const Plaintext<Scheme::BFV>& input2,
                         Ciphertext<Scheme::BFV>& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    multiply_plain(input1_, input2_,
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
         * @brief Performs in-place logical OR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        OR_inplace(Ciphertext<Scheme::BFV>& input1,
                   const Plaintext<Scheme::BFV>& input2,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

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
        __host__ void XOR(const Ciphertext<Scheme::BFV>& input1,
                          const Ciphertext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const Relinkey<Scheme::BFV>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
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
        XOR_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Ciphertext<Scheme::BFV>& input2,
                    const Relinkey<Scheme::BFV>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void XOR(const Ciphertext<Scheme::BFV>& input1,
                          const Plaintext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
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
        XOR_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Plaintext<Scheme::BFV>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

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
        __host__ void NAND(const Ciphertext<Scheme::BFV>& input1,
                           const Ciphertext<Scheme::BFV>& input2,
                           Ciphertext<Scheme::BFV>& output,
                           const Relinkey<Scheme::BFV>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

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
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
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
         * @brief Performs in-place logical NAND on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext<Scheme::BFV>& input1,
                     const Ciphertext<Scheme::BFV>& input2,
                     const Relinkey<Scheme::BFV>& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void NAND(const Ciphertext<Scheme::BFV>& input1,
                           const Plaintext<Scheme::BFV>& input2,
                           Ciphertext<Scheme::BFV>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    multiply_plain(input1_, input2_, output_,
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
         * @brief Performs in-place logical NAND on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext<Scheme::BFV>& input1,
                     const Plaintext<Scheme::BFV>& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

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
        __host__ void NOR(const Ciphertext<Scheme::BFV>& input1,
                          const Ciphertext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const Relinkey<Scheme::BFV>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
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
        NOR_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Ciphertext<Scheme::BFV>& input2,
                    const Relinkey<Scheme::BFV>& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void NOR(const Ciphertext<Scheme::BFV>& input1,
                          const Plaintext<Scheme::BFV>& input2,
                          Ciphertext<Scheme::BFV>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);

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
        NOR_inplace(Ciphertext<Scheme::BFV>& input1,
                    const Plaintext<Scheme::BFV>& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

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
        __host__ void XNOR(const Ciphertext<Scheme::BFV>& input1,
                           const Ciphertext<Scheme::BFV>& input2,
                           Ciphertext<Scheme::BFV>& output,
                           const Relinkey<Scheme::BFV>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

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
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
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
        XNOR_inplace(Ciphertext<Scheme::BFV>& input1,
                     const Ciphertext<Scheme::BFV>& input2,
                     const Relinkey<Scheme::BFV>& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);
            relin_key.validate_context(context_);

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
        __host__ void XNOR(const Ciphertext<Scheme::BFV>& input1,
                           const Plaintext<Scheme::BFV>& input2,
                           Ciphertext<Scheme::BFV>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::device)
                    .set_initial_location(true);

            Ciphertext<Scheme::BFV> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::BFV> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::BFV>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::BFV>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::BFV>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

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
         * @brief Performs in-place logical XNOR on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        XNOR_inplace(Ciphertext<Scheme::BFV>& input1,
                     const Plaintext<Scheme::BFV>& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            input1.validate_context(context_);
            input2.validate_context(context_);

            XNOR(input1, input2, input1, options);
        }

        __host__ void
        one_minus_cipher(const Ciphertext<Scheme::BFV>& input1,
                         Ciphertext<Scheme::BFV>& output,
                         const ExecutionOptions& options = ExecutionOptions());

        __host__ void one_minus_cipher_inplace(
            Ciphertext<Scheme::BFV>& input1,
            const ExecutionOptions& options = ExecutionOptions());

        // Encoded One
        DeviceVector<Data64> encoded_constant_one_;
    };

} // namespace heongpu

#endif // HEONGPU_BFV_OPERATOR_H
