// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_KEYGENERATOR_H
#define HEONGPU_BFV_KEYGENERATOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/secretkey.cuh>
#include <heongpu/host/bfv/publickey.cuh>
#include <heongpu/host/bfv/evaluationkey.cuh>

namespace heongpu
{

    /**
     * @brief HEKeyGenerator is responsible for generating various keys used in
     * homomorphic encryption, such as secret keys, public keys, relinearization
     * keys, Galois keys, and switch keys.
     *
     * The HEKeyGenerator class is initialized with encryption parameters and
     * provides methods to generate the keys required for different homomorphic
     * encryption schemes. It helps facilitate the encryption, decryption, and
     * key switching operations.
     *
     * Additionally, the class supports multiparty computation (MPC) by
     * implementing key generation protocols for collaborative schemes. These
     * include the generation of multiparty public keys, relinearization keys,
     * and Galois keys. The implementation is based on the techniques described
     * in the paper "Multiparty Homomorphic Encryption from
     * Ring-Learning-With-Errors".
     */
    template <> class HEKeyGenerator<Scheme::BFV>
    {
      public:
        /**
         * @brief Constructs a new HEKeyGenerator object with specified
         * parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ HEKeyGenerator(HEContext<Scheme::BFV> context);

        /**
         * @brief Generates an exact-weight ternary secret key using private
         * CSPRNG output.
         *
         * Legacy seed setters do not control secret-key sampling.
         * @throws std::runtime_error if private random generation fails.
         *
         * @param sk Reference to the Secretkey object where the generated
         * secret key will be stored.
         */
        __host__ void generate_secret_key(
            Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a public key using a secret key.
         *
         * @param pk Reference to the Publickey object where the generated
         * public key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * public key.
         */
        __host__ void generate_public_key(
            Publickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a relinearization key using a secret key.
         *
         * @param rk Reference to the Relinkey object where the generated
         * relinearization key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * relinearization key.
         */
        __host__ void
        generate_relin_key(Relinkey<Scheme::BFV>& rk,
                           Secretkey<Scheme::BFV>& sk,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            generate_bfv_relin_key(rk, sk, options);
        }

        /**
         * @brief Generates a Galois key using a secret key.
         *
         * @param gk Reference to the Galoiskey object where the generated
         * Galois key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * Galois key.
         */
        __host__ void generate_galois_key(
            Galoiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            generate_bfv_galois_key(gk, sk, options);
        }

        /**
         * @brief Generates a switch key for key switching between two secret
         * keys.
         *
         * @param swk Reference to the Switchkey object where the generated
         * switch key will be stored.
         * @param new_sk Reference to the new Secretkey object to switch to.
         * @param old_sk Reference to the old Secretkey object to switch from.
         */
        __host__ void generate_switch_key(
            Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
            Secretkey<Scheme::BFV>& old_sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            generate_bfv_switch_key(swk, new_sk, old_sk, options);
        }

        /**
         * @brief Returns the seed of the key generator.
         *
         * @return int Seed of the key generator.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_seed() const noexcept
        {
            return seed_;
        }

        /**
         * @brief Sets the seed of the key generator with new seed.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_seed(int new_seed)
        {
            seed_ = new_seed;
        }

        /**
         * @brief Returns the offset of the key generator (no effect).
         *
         * @return int Offset of the key generator.
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline int
        get_offset() const noexcept
        {
            return offset_;
        }

        /**
         * @brief Sets the offset of the key generator with new offset (no effect).
         */
        [[deprecated("Has no effect: random numbers come from "
                     "heongpu::RandomNumberGenerator.")]] inline void
        set_offset(int new_offset)
        {
            offset_ = new_offset;
        }

        HEKeyGenerator() = delete;
        HEKeyGenerator(const HEKeyGenerator& copy) = delete;
        HEKeyGenerator(HEKeyGenerator&& source) = delete;
        HEKeyGenerator& operator=(const HEKeyGenerator& assign) = delete;
        HEKeyGenerator& operator=(HEKeyGenerator&& assign) = delete;

        // Value-returning versions of the functions above. They give the
        // same results.

        /**
         * @brief Returns a new secret key; see generate_secret_key.
         */
        __host__ Secretkey<Scheme::BFV>
        secret_key(const ExecutionOptions& options = ExecutionOptions())
        {
            Secretkey<Scheme::BFV> output(context_);
            generate_secret_key(output, options);
            return output;
        }

        /**
         * @brief Returns a new Publickey of sk; see generate_public_key.
         */
        __host__ Publickey<Scheme::BFV>
        public_key(Secretkey<Scheme::BFV>& sk,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Publickey<Scheme::BFV> output(context_);
            generate_public_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Relinkey of sk; see generate_relin_key.
         */
        __host__ Relinkey<Scheme::BFV>
        relin_key(Secretkey<Scheme::BFV>& sk,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Relinkey<Scheme::BFV> output(context_);
            generate_relin_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Galoiskey of sk; see generate_galois_key.
         */
        __host__ Galoiskey<Scheme::BFV>
        galois_key(Secretkey<Scheme::BFV>& sk,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Galoiskey<Scheme::BFV> output(context_);
            generate_galois_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Galoiskey of sk for the given shifts; see
         * generate_galois_key.
         */
        __host__ Galoiskey<Scheme::BFV>
        galois_key(Secretkey<Scheme::BFV>& sk, std::vector<int>& shifts,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Galoiskey<Scheme::BFV> output(context_, shifts);
            generate_galois_key(output, sk, options);
            return output;
        }

      private:
        __host__ void generate_bfv_relin_key(Relinkey<Scheme::BFV>& rk,
                                             Secretkey<Scheme::BFV>& sk,
                                             const ExecutionOptions& options);

        __host__ void generate_bfv_galois_key(Galoiskey<Scheme::BFV>& gk,
                                              Secretkey<Scheme::BFV>& sk,
                                              const ExecutionOptions& options);

        __host__ void generate_bfv_switch_key(Switchkey<Scheme::BFV>& swk,
                                              Secretkey<Scheme::BFV>& new_sk,
                                              Secretkey<Scheme::BFV>& old_sk,
                                              const ExecutionOptions& options);

      private:
        HEContext<Scheme::BFV> context_;
        int seed_;
        int offset_; // Unused, see get_offset()

        RNGSeed new_seed_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_KEYGENERATOR_H
