// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_ENCODER_H
#define HEONGPU_CKKS_ENCODER_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpufft/fft.cuh"
#include <heongpu/kernel/encoding.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/plaintext.cuh>

namespace heongpu
{

    /**
     * @brief HEEncoder is responsible for encoding messages into plaintexts
     * suitable for homomorphic encryption.
     *
     * The HEEncoder class is initialized with encryption parameters and
     * provides methods to encode different types of messages (e.g., integers,
     * floating-point numbers, complex numbers) into plaintexts for BFV and CKKS
     * schemes. The class supports both synchronous and asynchronous encoding
     * operations, making it suitable for different homomorphic encryption
     * workflows.
     */
    template <> class HEEncoder<Scheme::CKKS>
    {
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;

      public:
        /**
         * @brief Constructs a new HEEncoder object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encode parameters.
         */
        __host__ HEEncoder(HEContext<Scheme::CKKS> context);

        /**
         * @brief Encodes a message of double values into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of double values representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<double>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if (type == encoding::slot)
            {
                if (message.size() > slot_count_)
                {
                    throw std::invalid_argument(
                        "Vector size can not be higher than slot count!");
                }
            }
            else
            {
                if (message.size() > static_cast<std::size_t>(context_->n))
                {
                    throw std::invalid_argument("Vector size can not be higher "
                                                "than polynomial degree!");
                }
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (type == encoding::slot)
                    {
                        encode_ckks(plain_, message, scale, options.stream_);
                    }
                    else
                    {
                        encode_ckks_coeff(plain_, message, scale,
                                          options.stream_);
                    }

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of double values into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of double values representing the message
         * to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<double>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if (type == encoding::slot)
            {
                if (message.size() > slot_count_)
                {
                    throw std::invalid_argument(
                        "Vector size can not be higher than slot count!");
                }
            }
            else
            {
                if (message.size() > static_cast<std::size_t>(context_->n))
                {
                    throw std::invalid_argument("Vector size can not be higher "
                                                "than polynomial degree!");
                }
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (type == encoding::slot)
                    {
                        encode_ckks(plain_, message, scale, options.stream_);
                    }
                    else
                    {
                        encode_ckks_coeff(plain_, message, scale,
                                          options.stream_);
                    }

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of complex numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of Complex64 representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<Complex64>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if (type == encoding::coefficient)
            {
                throw std::invalid_argument(
                    "Complex coefficient encoding is not supported for CKKS.");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of complex numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of Complex64 representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<Complex64>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if (type == encoding::coefficient)
            {
                throw std::invalid_argument(
                    "Complex coefficient encoding is not supported for CKKS.");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message into a plaintext at the given depth, so that
         * it can be used with ciphertexts of that depth without mod_drop.
         * Gives the same plaintext as encoding and then calling mod_drop depth
         * times.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Values to encode.
         * @param scale Scale of the encoding.
         * @param depth Number of dropped moduli, as Ciphertext::depth() returns
         * it; from 0 to the number of ciphertext moduli minus one.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<double>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            validate_depth(depth);
            encode(plain, message, scale, options, type);
            drop_to_depth(plain, depth, options);
        }

        /**
         * @brief Encodes a message into a plaintext at the given depth; see
         * the std::vector<double> version.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<double>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            validate_depth(depth);
            encode(plain, message, scale, options, type);
            drop_to_depth(plain, depth, options);
        }

        /**
         * @brief Encodes a message into a plaintext at the given depth; see
         * the std::vector<double> version.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<Complex64>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            validate_depth(depth);
            encode(plain, message, scale, options, type);
            drop_to_depth(plain, depth, options);
        }

        /**
         * @brief Encodes a message into a plaintext at the given depth; see
         * the std::vector<double> version.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<Complex64>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            validate_depth(depth);
            encode(plain, message, scale, options, type);
            drop_to_depth(plain, depth, options);
        }

        /**
         * @brief Encodes a message of single double numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Double representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain, const double& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if ((static_cast<int>(log2(fabs(message))) + 2) >=
                context_->total_coeff_bit_count)
            {
                throw std::invalid_argument("Encoded value is too large");
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of single int64_t numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message int64_t representing the message to be
         * encoded.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain, const std::int64_t& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                 context_->total_coeff_bit_count))
            {
                throw ParameterError(error_message(
                    "Scale out of bounds! The scale is ", scale,
                    " but it must be positive and smaller than 2^",
                    context_->total_coeff_bit_count, "."));
            }

            if ((static_cast<int>(log2(fabs(message))) + 2) >=
                context_->total_coeff_bit_count)
            {
                throw std::invalid_argument("Encoded value is too large");
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain_.plain_size_ = context_->n * context_->Q_size;
                    plain_.scheme_ = context_->scheme_;
                    plain_.context_id_ = context_->context_id_;
                    plain_.depth_ = 0;
                    plain_.scale_ = scale;
                    plain_.in_ntt_domain_ = true;
                    plain_.encoding_ = type;
                    plain_.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Decodes a plaintext into a vector of double values.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<double>& message,
               const Plaintext<Scheme::CKKS>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (plain_.encoding_ == encoding::coefficient)
                    {
                        decode_ckks_coeff(message, plain_, options.stream_);
                    }
                    else
                    {
                        decode_ckks(message, plain_, options.stream_);
                    }
                },
                options, false);
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of double values.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<double>& message,
               const Plaintext<Scheme::CKKS>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (plain_.encoding_ == encoding::coefficient)
                    {
                        decode_ckks_coeff(message, plain_, options.stream_);
                    }
                    else
                    {
                        decode_ckks(message, plain_, options.stream_);
                    }
                },
                options, false);
        }

        /**
         * @brief Decodes a plaintext into a vector of complex numbers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<Complex64>& message,
               const Plaintext<Scheme::CKKS>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (plain_.encoding_ == encoding::coefficient)
                    {
                        throw std::invalid_argument(
                            "Coefficient encoded CKKS plaintext can not be "
                            "decoded to complex slots.");
                    }
                    decode_ckks(message, plain_, options.stream_);
                },
                options, false);
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of complex numbers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<Complex64>& message,
               const Plaintext<Scheme::CKKS>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    if (plain_.encoding_ == encoding::coefficient)
                    {
                        throw std::invalid_argument(
                            "Coefficient encoded CKKS plaintext can not be "
                            "decoded to complex slots.");
                    }
                    decode_ckks(message, plain_, options.stream_);
                },
                options, false);
        }

        /**
         * @brief Returns the number of slots.
         *
         * @return int Number of slots.
         */
        inline int slot_count() const noexcept { return slot_count_; }

        /**
         * @brief Encodes a message of double values at the top level with its
         * scale, in rescale_mode::automatic; see
         * HEContextImpl<Scheme::CKKS>::set_rescale_mode. Throws StateError in
         * rescale_mode::manual, which needs a scale.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<double>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(0), options, type);
        }

        /**
         * @brief Encodes a message of double values at the given depth with the
         * scale of its level, in rescale_mode::automatic, for ciphertexts of
         * that depth; see encode(plain, message, scale, depth).
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<double>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(depth), depth, options, type);
        }

        /**
         * @brief Encodes a message of double values at the top level with its
         * scale, in rescale_mode::automatic; see
         * HEContextImpl<Scheme::CKKS>::set_rescale_mode. Throws StateError in
         * rescale_mode::manual, which needs a scale.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<double>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(0), options, type);
        }

        /**
         * @brief Encodes a message of double values at the given depth with the
         * scale of its level, in rescale_mode::automatic, for ciphertexts of
         * that depth; see encode(plain, message, scale, depth).
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<double>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(depth), depth, options, type);
        }

        /**
         * @brief Encodes a message of complex values at the top level with its
         * scale, in rescale_mode::automatic; see
         * HEContextImpl<Scheme::CKKS>::set_rescale_mode. Throws StateError in
         * rescale_mode::manual, which needs a scale.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<Complex64>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(0), options, type);
        }

        /**
         * @brief Encodes a message of complex values at the given depth with
         * the scale of its level, in rescale_mode::automatic, for ciphertexts
         * of that depth; see encode(plain, message, scale, depth).
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<Complex64>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(depth), depth, options, type);
        }

        /**
         * @brief Encodes a message of complex values at the top level with its
         * scale, in rescale_mode::automatic; see
         * HEContextImpl<Scheme::CKKS>::set_rescale_mode. Throws StateError in
         * rescale_mode::manual, which needs a scale.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<Complex64>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(0), options, type);
        }

        /**
         * @brief Encodes a message of complex values at the given depth with
         * the scale of its level, in rescale_mode::automatic, for ciphertexts
         * of that depth; see encode(plain, message, scale, depth).
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<Complex64>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            encode(plain, message, level_scale(depth), depth, options, type);
        }

        HEEncoder() = default;
        HEEncoder(const HEEncoder& copy) = default;
        HEEncoder(HEEncoder&& source) = default;
        HEEncoder& operator=(const HEEncoder& assign) = default;
        HEEncoder& operator=(HEEncoder&& assign) = default;

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns message encoded into a new plaintext; see
         * encode(plain, message, scale).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<double>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, scale, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the given
         * depth; see encode(plain, message, scale, depth).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<double>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, scale, depth, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext; see
         * encode(plain, message, scale).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<Complex64>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, scale, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the given
         * depth; see encode(plain, message, scale, depth).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<Complex64>& message, double scale, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, scale, depth, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the top level
         * with its scale, in rescale_mode::automatic; see encode(plain,
         * message).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<double>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the given
         * depth with the scale of its level, in rescale_mode::automatic; see
         * encode(plain, message, depth).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<double>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, depth, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the top level
         * with its scale, in rescale_mode::automatic; see encode(plain,
         * message).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<Complex64>& message,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, options, type);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext at the given
         * depth with the scale of its level, in rescale_mode::automatic; see
         * encode(plain, message, depth).
         */
        __host__ Plaintext<Scheme::CKKS>
        encode(const std::vector<Complex64>& message, int depth,
               const ExecutionOptions& options = ExecutionOptions(),
               encoding type = encoding::slot)
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            encode(output, message, depth, options, type);
            return output;
        }

        /**
         * @brief Returns the values decoded from plain; see decode(message,
         * plain).
         */
        __host__ std::vector<double>
        decode(const Plaintext<Scheme::CKKS>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            std::vector<double> message;
            decode(message, plain, options);
            return message;
        }

      private:
        // The scale of the level of depth in rescale_mode::automatic.
        double level_scale(int depth) const
        {
            validate_depth(depth);
            if (context_->rescale_mode_ != rescale_mode::automatic)
            {
                throw StateError("Encoding without a scale needs the automatic "
                                 "rescale mode! Pass a scale, or set the mode "
                                 "with set_rescale_mode.");
            }
            return context_->level_scales_[context_->Q_size - 1 - depth];
        }

        void validate_depth(int depth) const
        {
            if (depth < 0 || depth >= context_->Q_size)
            {
                throw ParameterError(error_message(
                    "Invalid depth ", depth, "! It must be between 0 and ",
                    context_->Q_size - 1, "."));
            }
        }

        // Keeps the moduli of the given depth, as mod_drop does.
        void drop_to_depth(Plaintext<Scheme::CKKS>& plain, int depth,
                           const ExecutionOptions& options)
        {
            if (depth == 0)
            {
                return;
            }
            const int size = context_->n * (context_->Q_size - depth);
            if (plain.storage_type_ == storage_type::device)
            {
                DeviceVector<Data64> kept(size, options.stream_);
                HEONGPU_CUDA_CHECK(
                    cudaMemcpyAsync(kept.data(), plain.device_locations_.data(),
                                    size * sizeof(Data64),
                                    cudaMemcpyDeviceToDevice, options.stream_));
                plain.memory_set(std::move(kept));
            }
            else
            {
                plain.host_locations_.resize(size);
            }
            plain.plain_size_ = size;
            plain.depth_ = depth;
        }

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::vector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const HostVector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks_coeff(Plaintext<Scheme::CKKS>& plain,
                                        const std::vector<double>& message,
                                        const double scale,
                                        const cudaStream_t stream);

        __host__ void encode_ckks_coeff(Plaintext<Scheme::CKKS>& plain,
                                        const HostVector<double>& message,
                                        const double scale,
                                        const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::vector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const HostVector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const double& message, const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::int64_t& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<double>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<double>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        __host__ void decode_ckks_coeff(std::vector<double>& message,
                                        Plaintext<Scheme::CKKS>& plain,
                                        const cudaStream_t stream);

        __host__ void decode_ckks_coeff(HostVector<double>& message,
                                        Plaintext<Scheme::CKKS>& plain,
                                        const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<Complex64>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<Complex64>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

      private:
        HEContext<Scheme::CKKS> context_;
        int slot_count_;

        double two_pow_64;
        int log_slot_count_;
        int fft_length;
        Complex64 special_root;

        std::shared_ptr<DeviceVector<Complex64>> special_fft_roots_table_;
        std::shared_ptr<DeviceVector<Complex64>> special_ifft_roots_table_;
        std::shared_ptr<DeviceVector<int>> reverse_order;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_ENCODER_H
