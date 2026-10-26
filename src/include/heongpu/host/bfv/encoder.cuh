// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_ENCODER_H
#define HEONGPU_BFV_ENCODER_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpufft/fft.cuh"
#include <heongpu/kernel/encoding.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/plaintext.cuh>

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
    template <> class HEEncoder<Scheme::BFV>
    {
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;

      public:
        /**
         * @brief Constructs a new HEEncoder object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encode parameters.
         */
        __host__ HEEncoder(HEContext<Scheme::BFV> context);

        /**
         * @brief Encodes a message into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of unsigned 64-bit integers representing the
         * message to be encoded.
         * @param scale Parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::BFV>& plain,
               const std::vector<uint64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if (message.size() > slot_count())
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                {
                    encode_bfv(plain_, message, options.stream_);

                    plain.plain_size_ = context_->n;
                    plain.scheme_ = context_->scheme_;
                    plain.context_id_ = context_->context_id_;
                    plain.in_ntt_domain_ = false;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Encodes a message of signed 64-bit integers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of signed 64-bit integers representing the
         * message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::BFV>& plain,
               const std::vector<int64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if (message.size() > slot_count())
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                {
                    encode_bfv(plain_, message, options.stream_);

                    plain.plain_size_ = context_->n;
                    plain.scheme_ = context_->scheme_;
                    plain.context_id_ = context_->context_id_;
                    plain.in_ntt_domain_ = false;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Encodes a message into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of unsigned 64-bit integers representing
         * the message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::BFV>& plain,
               const HostVector<uint64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if (message.size() > slot_count())
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                {
                    encode_bfv(plain_, message, options.stream_);

                    plain.plain_size_ = context_->n;
                    plain.scheme_ = context_->scheme_;
                    plain.context_id_ = context_->context_id_;
                    plain.in_ntt_domain_ = false;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Encodes a message of signed 64-bit integers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of signed 64-bit integers representing the
         * message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::BFV>& plain,
               const HostVector<int64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if (message.size() > slot_count())
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                {
                    encode_bfv(plain_, message, options.stream_);

                    plain.plain_size_ = context_->n;
                    plain.scheme_ = context_->scheme_;
                    plain.context_id_ = context_->context_id_;
                    plain.in_ntt_domain_ = false;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Decodes a plaintext into a vector of unsigned 64-bit integers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<uint64_t>& message, Plaintext<Scheme::BFV>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                { decode_bfv(message, plain_, options.stream_); },
                options, false);
        }

        /**
         * @brief Decodes a plaintext into a vector of signed 64-bit integers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<int64_t>& message, Plaintext<Scheme::BFV>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                { decode_bfv(message, plain_, options.stream_); },
                options, false);
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of unsigned 64-bit
         * integers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<uint64_t>& message, Plaintext<Scheme::BFV>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                { decode_bfv(message, plain_, options.stream_); },
                options, false);
        }

        /**
         * @brief Decodes a plaintext into a HostVector of signed 64-bit
         * integers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<int64_t>& message, Plaintext<Scheme::BFV>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            plain.validate_context(context_);

            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::BFV>& plain_)
                { decode_bfv(message, plain_, options.stream_); },
                options, false);
        }

        /**
         * @brief Returns the number of slots.
         *
         * @return int Number of slots.
         */
        inline int slot_count() const noexcept
        {
            return context_ ? context_->n : 0;
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
         * encode(plain, message).
         */
        __host__ Plaintext<Scheme::BFV>
        encode(const std::vector<uint64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            Plaintext<Scheme::BFV> output(context_, options);
            encode(output, message, options);
            return output;
        }

        /**
         * @brief Returns message encoded into a new plaintext; see
         * encode(plain, message).
         */
        __host__ Plaintext<Scheme::BFV>
        encode(const std::vector<int64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            Plaintext<Scheme::BFV> output(context_, options);
            encode(output, message, options);
            return output;
        }

        /**
         * @brief Returns the values decoded from plain; see decode(message,
         * plain).
         */
        __host__ std::vector<uint64_t>
        decode(Plaintext<Scheme::BFV>& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            std::vector<uint64_t> message;
            decode(message, plain, options);
            return message;
        }

      private:
        __host__ void encode_bfv(Plaintext<Scheme::BFV>& plain,
                                 const std::vector<uint64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext<Scheme::BFV>& plain,
                                 const std::vector<int64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext<Scheme::BFV>& plain,
                                 const HostVector<uint64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext<Scheme::BFV>& plain,
                                 const HostVector<int64_t>& message,
                                 const cudaStream_t stream);

        __host__ void decode_bfv(std::vector<uint64_t>& message,
                                 Plaintext<Scheme::BFV>& plain,
                                 const cudaStream_t stream);

        __host__ void decode_bfv(std::vector<int64_t>& message,
                                 Plaintext<Scheme::BFV>& plain,
                                 const cudaStream_t stream);

        __host__ void decode_bfv(HostVector<uint64_t>& message,
                                 Plaintext<Scheme::BFV>& plain,
                                 const cudaStream_t stream);

        __host__ void decode_bfv(HostVector<int64_t>& message,
                                 Plaintext<Scheme::BFV>& plain,
                                 const cudaStream_t stream);

      private:
        HEContext<Scheme::BFV> context_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_ENCODER_H
