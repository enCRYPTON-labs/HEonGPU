// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_DECRYPTOR_H
#define HEONGPU_CKKS_DECRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/addition.cuh>
#include <heongpu/kernel/decryption.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/secretkey.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>

namespace heongpu
{

    /**
     * @brief HEDecryptor is responsible for decrypting ciphertexts back into
     * plaintexts using a secret key in homomorphic encryption schemes.
     *
     * The HEDecryptor class is initialized with encryption parameters and a
     * secret key. It provides methods to decrypt ciphertexts into plaintexts
     * for BFV and CKKS schemes. The class also supports noise budget
     * calculation, which is essential for understanding the remaining "noise"
     * tolerance in a given ciphertext.
     *
     * Additionally, the class includes methods for multiparty computation (MPC)
     * scenarios. These methods enable partial decryption by multiple
     * participants and the fusion of these partial decryptions into a fully
     * decrypted plaintext.
     */
    template <> class HEDecryptor<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEDecryptor object with specified parameters
         * and secret key.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param secret_key Reference to the Secretkey object used for
         * decryption.
         */
        __host__ HEDecryptor(HEContext<Scheme::CKKS> context,
                             Secretkey<Scheme::CKKS>& secret_key);

        /**
         * @brief Decrypts a ciphertext into a plaintext, automatically
         * determining the scheme type.
         *
         * @param plaintext Plaintext object where the result of the decryption
         * will be stored.
         * @param ciphertext Ciphertext object to be decrypted.
         */
        __host__ void
        decrypt(Plaintext<Scheme::CKKS>& plaintext,
                Ciphertext<Scheme::CKKS>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            ciphertext.validate_context(context_);
            if (ciphertext.cipher_size_ != 2 || !ciphertext.in_ntt_domain_)
            {
                throw std::invalid_argument(
                    "CKKS decryption requires a size-2 NTT ciphertext!");
            }
            input_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::CKKS>& ciphertext_)
                {
                    output_storage_manager(
                        plaintext,
                        [&](Plaintext<Scheme::CKKS>& plaintext_)
                        {
                            decrypt_ckks(plaintext_, ciphertext,
                                         options.stream_);

                            plaintext.plain_size_ =
                                context_->n *
                                (context_->Q_size - ciphertext.depth_);
                            plaintext.scheme_ = context_->scheme_;
                            plaintext.context_id_ = context_->context_id_;
                            plaintext.depth_ = ciphertext.depth_;
                            plaintext.scale_ = ciphertext.scale_;
                            plaintext.in_ntt_domain_ = true;
                            plaintext.encoding_ = ciphertext.encoding_;
                            plaintext.plaintext_generated_ = true;
                        },
                        options);
                },
                options, false);
        }

        /**
         * @brief Returns the seed of the decryptor.
         *
         * @return int Seed of the decryptor.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_seed() const noexcept
        {
            return seed_;
        }

        /**
         * @brief Sets the seed of the decryptor with new seed.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_seed(int new_seed)
        {
            seed_ = new_seed;
        }

        /**
         * @brief Returns the offset of the decryptor (no effect).
         *
         * @return int Offset of the decryptor.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_offset() const noexcept
        {
            return offset_;
        }

        /**
         * @brief Sets the offset of the decryptor with new offset (no effect).
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_offset(int new_offset)
        {
            offset_ = new_offset;
        }

        HEDecryptor() = default;
        HEDecryptor(const HEDecryptor& copy) = default;
        HEDecryptor(HEDecryptor&& source) = default;
        HEDecryptor& operator=(const HEDecryptor& assign) = default;
        HEDecryptor& operator=(HEDecryptor&& assign) = default;

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns ciphertext decrypted into a new plaintext; see
         * decrypt(plaintext, ciphertext).
         */
        __host__ Plaintext<Scheme::CKKS>
        decrypt(Ciphertext<Scheme::CKKS>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            Plaintext<Scheme::CKKS> output(context_, options);
            decrypt(output, ciphertext, options);
            return output;
        }

      private:
        __host__ void decrypt_ckks(Plaintext<Scheme::CKKS>& plaintext,
                                   Ciphertext<Scheme::CKKS>& ciphertext,
                                   const cudaStream_t stream);

        __host__ void
        partial_decrypt_ckks(Ciphertext<Scheme::CKKS>& ciphertext,
                             Secretkey<Scheme::CKKS>& sk,
                             Ciphertext<Scheme::CKKS>& partial_ciphertext,
                             const cudaStream_t stream);

        __host__ void
        decrypt_fusion_ckks(std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
                            Plaintext<Scheme::CKKS>& plaintext,
                            const cudaStream_t stream);

      private:
        HEContext<Scheme::CKKS> context_;
        int seed_;
        int offset_; // Unused, see get_offset()

        DeviceVector<Data64> secret_key_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_DECRYPTOR_H
