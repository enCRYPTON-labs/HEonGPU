// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_DECRYPTOR_H
#define HEONGPU_BFV_DECRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/addition.cuh>
#include <heongpu/kernel/decryption.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/secretkey.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/ciphertext.cuh>

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
    template <> class HEDecryptor<Scheme::BFV>
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
        __host__ HEDecryptor(HEContext<Scheme::BFV> context,
                             Secretkey<Scheme::BFV>& secret_key);

        /**
         * @brief Decrypts a ciphertext into a plaintext, automatically
         * determining the scheme type.
         *
         * @param plaintext Plaintext object where the result of the decryption
         * will be stored.
         * @param ciphertext Ciphertext object to be decrypted.
         */
        __host__ void
        decrypt(Plaintext<Scheme::BFV>& plaintext,
                Ciphertext<Scheme::BFV>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            ciphertext.validate_context(context_);

            input_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::BFV>& ciphertext_)
                {
                    output_storage_manager(
                        plaintext,
                        [&](Plaintext<Scheme::BFV>& plaintext_)
                        {
                            decrypt_bfv(plaintext_, ciphertext_,
                                        options.stream_);

                            plaintext.plain_size_ = context_->n;
                            plaintext.scheme_ = context_->scheme_;
                            plaintext.context_id_ = context_->context_id_;
                            plaintext.in_ntt_domain_ = false;
                        },
                        options);
                },
                options, false);
        }

        /**
         * @brief Calculates the remainder of the noise budget in a ciphertext.
         *
         * @param ciphertext Ciphertext object for which the remaining noise
         * budget is calculated.
         * @return int The remainder of the noise budget in the ciphertext.
         */
        __host__ int remainder_noise_budget(
            Ciphertext<Scheme::BFV>& ciphertext,
            const ExecutionOptions& options = ExecutionOptions())
        {
            return noise_budget_calculation(ciphertext, options);
        }

        /**
         * @brief Returns the seed of the decryptor.
         *
         * @return int Seed of the decryptor.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the seed of the decryptor with new seed.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the decryptor(curand).
         *
         * @return int Offset of the decryptor.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the decryptor with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

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
        __host__ Plaintext<Scheme::BFV>
        decrypt(Ciphertext<Scheme::BFV>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            Plaintext<Scheme::BFV> output(context_, options);
            decrypt(output, ciphertext, options);
            return output;
        }

      private:
        __host__ void decrypt_bfv(Plaintext<Scheme::BFV>& plaintext,
                                  Ciphertext<Scheme::BFV>& ciphertext,
                                  const cudaStream_t stream);

        __host__ void decryptx3_bfv(Plaintext<Scheme::BFV>& plaintext,
                                    Ciphertext<Scheme::BFV>& ciphertext,
                                    const cudaStream_t stream);

        __host__ int noise_budget_calculation(
            Ciphertext<Scheme::BFV>& ciphertext,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        partial_decrypt_bfv(Ciphertext<Scheme::BFV>& ciphertext,
                            Secretkey<Scheme::BFV>& sk,
                            Ciphertext<Scheme::BFV>& partial_ciphertext,
                            const cudaStream_t stream);

        __host__ void
        decrypt_fusion_bfv(std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
                           Plaintext<Scheme::BFV>& plaintext,
                           const cudaStream_t stream);

      private:
        HEContext<Scheme::BFV> context_;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        DeviceVector<Data64> secret_key_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_DECRYPTOR_H
