// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_ENCRYPTOR_H
#define HEONGPU_CKKS_ENCRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/encryption.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/publickey.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>

namespace heongpu
{

    /**
     * @brief HEEncryptor is responsible for encrypting plaintexts into
     * ciphertexts using public keys in homomorphic encryption schemes.
     *
     * The HEEncryptor class is initialized with encryption parameters and a
     * public key. It provides methods to encrypt plaintext data into
     * ciphertexts for BFV and CKKS schemes. The class supports both synchronous
     * and asynchronous encryption operations.
     */
    template <> class HEEncryptor<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEEncryptor object with specified parameters
         * and public key.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param public_key Reference to the Publickey object used for
         * encryption.
         */
        __host__ HEEncryptor(HEContext<Scheme::CKKS> context,
                             Publickey<Scheme::CKKS>& public_key);

        /**
         * @brief Encrypts a plaintext into a ciphertext, automatically
         * determining the scheme type.
         *
         * @param ciphertext Ciphertext object where the result of the
         * encryption will be stored.
         * @param plaintext Plaintext object to be encrypted.
         */
        __host__ void
        encrypt(Ciphertext<Scheme::CKKS>& ciphertext,
                Plaintext<Scheme::CKKS>& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            plaintext.validate_context(context_);

            if (plaintext.size() < (context_->n * context_->Q_size))
            {
                throw std::invalid_argument("Invalid plaintext size.");
            }

            if (plaintext.depth() != 0)
            {
                throw std::invalid_argument(
                    "Invalid plaintext depth must be zero.");
            }

            input_storage_manager(
                plaintext,
                [&](Plaintext<Scheme::CKKS>& plaintext_)
                {
                    output_storage_manager(
                        ciphertext,
                        [&](Ciphertext<Scheme::CKKS>& ciphertext_)
                        {
                            encrypt_ckks(ciphertext_, plaintext,
                                         options.stream_);

                            ciphertext.scheme_ = context_->scheme_;
                            ciphertext.context_id_ = context_->context_id_;
                            ciphertext.ring_size_ = context_->n;
                            ciphertext.coeff_modulus_count_ = context_->Q_size;
                            ciphertext.cipher_size_ = 2;
                            ciphertext.depth_ = 0;
                            ciphertext.in_ntt_domain_ = true;
                            ciphertext.encoding_ = plaintext.encoding_;
                            ciphertext.scale_ = plaintext.scale_;
                            ciphertext.rescale_required_ = false;
                            ciphertext.relinearization_required_ = false;
                            ciphertext.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, false);
        }

        /**
         * @brief Returns the seed of the encryptor.
         *
         * @return int Seed of the encryptor.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the seed of the encryptor with new seed.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the encryptor(curand).
         *
         * @return int Offset of the encryptor.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the encryptor with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

        HEEncryptor() = default;
        HEEncryptor(const HEEncryptor& copy) = default;
        HEEncryptor(HEEncryptor&& source) = default;
        HEEncryptor& operator=(const HEEncryptor& assign) = default;
        HEEncryptor& operator=(HEEncryptor&& assign) = default;

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns plaintext encrypted into a new ciphertext; see
         * encrypt(ciphertext, plaintext).
         */
        __host__ Ciphertext<Scheme::CKKS>
        encrypt(Plaintext<Scheme::CKKS>& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::CKKS> output(context_, options);
            encrypt(output, plaintext, options);
            return output;
        }

      private:
        __host__ void encrypt_ckks(Ciphertext<Scheme::CKKS>& ciphertext,
                                   Plaintext<Scheme::CKKS>& plaintext,
                                   const cudaStream_t stream);

      private:
        HEContext<Scheme::CKKS> context_;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        DeviceVector<Data64> public_key_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_ENCRYPTOR_H
