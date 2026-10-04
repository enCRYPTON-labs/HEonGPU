// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_KEYGENERATOR_H
#define HEONGPU_TFHE_KEYGENERATOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/host/tfhe/context.cuh>
#include <heongpu/host/tfhe/secretkey.cuh>
#include <heongpu/host/tfhe/evaluationkey.cuh>

namespace heongpu
{
    template <> class HEKeyGenerator<Scheme::TFHE>
    {
      public:
        /**
         * @brief Initializes a key generator for the TFHE scheme.
         *
         * Sets up internal state using the given encryption context.
         *
         * @param context TFHE encryption context.
         */
        __host__ HEKeyGenerator(HEContext<Scheme::TFHE> context);

        /**
         * @brief Generates independent binary LWE and TLWE keys using private
         * CSPRNG output.
         * @throws std::runtime_error if private random generation fails.
         *
         * Writes the result to the provided `sk` object.
         *
         * @param sk Output secret key.
         * @param options Optional CUDA execution settings.
         */
        __host__ void generate_secret_key(
            Secretkey<Scheme::TFHE>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a TFHE bootstrapping key and key switching key under
         * the Bootstrappingkey.
         *
         * Uses the given secret key and writes the result to `bk`.
         *
         * @param bk Output bootstrapping key.
         * @param sk Input secret key.
         * @param options Optional CUDA execution settings.
         */
        __host__ void generate_bootstrapping_key(
            Bootstrappingkey<Scheme::TFHE>& bk, Secretkey<Scheme::TFHE>& sk,
            const ExecutionOptions& options = ExecutionOptions());

      private:
        HEContext<Scheme::TFHE> context_;
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_KEYGENERATOR_H
