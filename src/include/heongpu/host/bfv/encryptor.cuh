// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_ENCRYPTOR_H
#define HEONGPU_BFV_ENCRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/encryption.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/publickey.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/ciphertext.cuh>

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
    template <> class HEEncryptor<Scheme::BFV>
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
        __host__ HEEncryptor(HEContext<Scheme::BFV> context,
                             Publickey<Scheme::BFV>& public_key);

        /**
         * @brief Encrypts a plaintext into a ciphertext, automatically
         * determining the scheme type.
         *
         * @param ciphertext Ciphertext object where the result of the
         * encryption will be stored.
         * @param plaintext Plaintext object to be encrypted.
         */
        __host__ void
        encrypt(Ciphertext<Scheme::BFV>& ciphertext,
                Plaintext<Scheme::BFV>& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            plaintext.validate_context(context_);

            if (plaintext.size() < context_->n)
            {
                throw std::invalid_argument("Invalid plaintext size.");
            }

            input_storage_manager(
                plaintext,
                [&](Plaintext<Scheme::BFV>& plaintext_)
                {
                    output_storage_manager(
                        ciphertext,
                        [&](Ciphertext<Scheme::BFV>& ciphertext_)
                        {
                            encrypt_bfv(ciphertext_, plaintext_,
                                        options.stream_);

                            ciphertext.scheme_ = context_->scheme_;
                            ciphertext.context_id_ = context_->context_id_;
                            ciphertext.ring_size_ = context_->n;
                            ciphertext.coeff_modulus_count_ = context_->Q_size;
                            ciphertext.cipher_size_ = 2;
                            ciphertext.in_ntt_domain_ = false;
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
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_seed() const noexcept
        {
            return seed_;
        }

        /**
         * @brief Sets the seed of the encryptor with new seed.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_seed(int new_seed)
        {
            seed_ = new_seed;
        }

        /**
         * @brief Returns the offset of the encryptor (no effect).
         *
         * @return int Offset of the encryptor.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_offset() const noexcept
        {
            return offset_;
        }

        /**
         * @brief Sets the offset of the encryptor with new offset (no effect).
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_offset(int new_offset)
        {
            offset_ = new_offset;
        }

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
        __host__ Ciphertext<Scheme::BFV>
        encrypt(Plaintext<Scheme::BFV>& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            Ciphertext<Scheme::BFV> output(context_, options);
            encrypt(output, plaintext, options);
            return output;
        }

      private:
        __host__ void encrypt_bfv(Ciphertext<Scheme::BFV>& ciphertext,
                                  Plaintext<Scheme::BFV>& plaintext,
                                  const cudaStream_t stream);

      private:
        HEContext<Scheme::BFV> context_;
        int seed_;
        int offset_; // Unused, see get_offset()

        DeviceVector<Data64> public_key_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_ENCRYPTOR_H
