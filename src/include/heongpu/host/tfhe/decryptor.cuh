// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_DECRYPTOR_H
#define HEONGPU_TFHE_DECRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/decryption.cuh>
#include <heongpu/host/tfhe/context.cuh>
#include <heongpu/host/tfhe/secretkey.cuh>
#include <heongpu/host/tfhe/ciphertext.cuh>

namespace heongpu
{
    template <> class HEDecryptor<Scheme::TFHE>
    {
      public:
        /**
         * @brief Constructs a new HEDecryptor object for performing decryption
         *        operations in the TFHE scheme.
         *
         * This constructor initializes the decryption context with the provided
         * encryption parameters and the associated secret key. It enables
         * decryption of ciphertexts that were previously encrypted under the
         * same parameters and key.
         *
         * @param context Reference to the HEContext object containing
         * encryption parameters for the TFHE scheme.
         * @param secret_key Reference to the Secretkey object associated with
         *                   the TFHE encryption scheme.
         */
        __host__ HEDecryptor(HEContext<Scheme::TFHE> context,
                             const Secretkey<Scheme::TFHE>& secret_key);

        /**
         * @brief Decrypts a TFHE ciphertext into a vector of boolean messages.
         *
         * This function performs decryption of a given ciphertext using the
         * TFHE scheme and writes the result into a boolean vector. Internally,
         * it invokes the appropriate decryption kernel via the input storage
         * manager, which handles any memory transfer or layout preparation
         * needed for GPU decryption.
         *
         * @param ciphertext Reference to the ciphertext to be decrypted.
         * @param messages Reference to a vector that will hold the decrypted
         * boolean values.
         * @param options Optional execution options, including the CUDA stream
         * to use. Defaults to `ExecutionOptions()`.
         */
        __host__ void
        decrypt(const Ciphertext<Scheme::TFHE>& ciphertext,
                std::vector<bool>& messages,
                const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::TFHE>& ciphertext_)
                { decrypt_lwe(messages, ciphertext_, options.stream_); },
                options, false);
        }

      private:
        __host__ void decrypt_lwe(std::vector<bool>& messages,
                                  Ciphertext<Scheme::TFHE>& ciphertext,
                                  const cudaStream_t stream);

      private:
        HEContext<Scheme::TFHE> context_;

        DeviceVector<int32_t> lwe_key_device_location_;
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_DECRYPTOR_H
