// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_ENCRYPTOR_H
#define HEONGPU_TFHE_ENCRYPTOR_H

#include <limits>

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/encryption.cuh>
#include <heongpu/host/tfhe/context.cuh>
#include <heongpu/host/tfhe/secretkey.cuh>
#include <heongpu/host/tfhe/ciphertext.cuh>

namespace heongpu
{
    template <> class HEEncryptor<Scheme::TFHE>
    {
      public:
        /**
         * @brief Constructs a new HEEncryptor object for encrypting messages
         *        under the TFHE scheme.
         *
         * This constructor sets up the encryption context with the specified
         * TFHE parameters and secret key. It enables symmetric encryption of
         * boolean messages into ciphertexts that can later be decrypted by
         * the corresponding decryptor.
         *
         * @param context Reference to the HEContext object containing
         * encryption parameters for the TFHE scheme.
         * @param secret_key Reference to the Secretkey object used for
         * encryption.
         */
        __host__ HEEncryptor(HEContext<Scheme::TFHE> context,
                             Secretkey<Scheme::TFHE>& secret_key);

        /**
         * @brief Encrypts a vector of boolean messages into a TFHE ciphertext.
         *
         * This function performs symmetric encryption by encoding the input
         * boolean messages into Torus format and encrypting them using the
         * secret key.
         *
         * @param ciphertext Output ciphertext where the encrypted result will
         * be stored.
         * @param messages Vector of boolean values representing the plaintext
         * to encrypt.
         */
        __host__ void
        encrypt(Ciphertext<Scheme::TFHE>& ciphertext,
                const std::vector<bool>& messages,
                const ExecutionOptions& options = ExecutionOptions())
        {
            int32_t one_over_8 = encode_to_torus32(1, 8);

            if (messages.size() >
                static_cast<size_t>(std::numeric_limits<int>::max() /
                                    context_->n_))
                throw std::invalid_argument("Too many TFHE messages!");
            Ciphertext<Scheme::TFHE> candidate(context_);
            candidate.shape_ = static_cast<int>(messages.size());

            std::vector<int32_t> encoded_messages;
            encoded_messages.reserve(candidate.shape_);
            for (bool bit : messages)
            {
                encoded_messages.push_back(bit ? one_over_8 : -one_over_8);
            }

            output_storage_manager(
                candidate,
                [&](Ciphertext<Scheme::TFHE>& ciphertext_)
                {
                    encrypt_lwe_symmetric(candidate, encoded_messages,
                                          options.stream_);

                    candidate.n_ = context_->n_;
                    candidate.alpha_min_ = alpha_min_;
                    candidate.alpha_max_ = alpha_max_;
                    candidate.ciphertext_generated_ = true;
                },
                options);
            ciphertext = std::move(candidate);
        }

        __host__ ~HEEncryptor() = default;

      private:
        __host__ void
        encrypt_lwe_symmetric(Ciphertext<Scheme::TFHE>& ciphertext,
                              std::vector<int32_t>& messages,
                              const cudaStream_t stream);

        __host__ int32_t encode_to_torus32(uint32_t mu, uint32_t m_size);

      private:
        HEContext<Scheme::TFHE> context_;

        double alpha_min_;
        double alpha_max_;

        DeviceVector<int32_t> lwe_key_device_location_;
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_ENCRYPTOR_H
