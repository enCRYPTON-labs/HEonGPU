// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_KEYGENERATOR_H
#define HEONGPU_CKKS_KEYGENERATOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/keygeneration.cuh>
#include <heongpu/kernel/switchkey.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/secretkey.cuh>
#include <heongpu/host/ckks/publickey.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>

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
    template <> class HEKeyGenerator<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEKeyGenerator object with specified
         * parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ HEKeyGenerator(HEContext<Scheme::CKKS> context);

        /**
         * @brief Generates a secret key using cryptographic private randomness.
         * The legacy seed value does not control secret key sampling.
         * Throws if the cryptographic random generator fails.
         *
         * @param sk Reference to the Secretkey object where the generated
         * secret key will be stored.
         */
        __host__ void generate_secret_key(
            Secretkey<Scheme::CKKS>& sk,
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
            Publickey<Scheme::CKKS>& pk, Secretkey<Scheme::CKKS>& sk,
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
        generate_relin_key(Relinkey<Scheme::CKKS>& rk,
                           Secretkey<Scheme::CKKS>& sk,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            generate_ckks_relin_key(rk, sk, options);
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
            Galoiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            generate_ckks_galois_key(gk, sk, options);
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
            Switchkey<Scheme::CKKS>& swk, Secretkey<Scheme::CKKS>& new_sk,
            Secretkey<Scheme::CKKS>& old_sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            generate_ckks_switch_key(swk, new_sk, old_sk, options);
        }

        /**
         * @brief Returns the legacy seed value retained for API compatibility.
         * This value is independent of secret key randomness.
         *
         * @return int Seed of the key generator.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the legacy seed value retained for API compatibility.
         * This does not make secret key generation deterministic.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the key generator(curand).
         *
         * @return int Offset of the key generator.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the key generator with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

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
        __host__ Secretkey<Scheme::CKKS>
        secret_key(const ExecutionOptions& options = ExecutionOptions())
        {
            Secretkey<Scheme::CKKS> output(context_);
            generate_secret_key(output, options);
            return output;
        }

        /**
         * @brief Returns a new Publickey of sk; see generate_public_key.
         */
        __host__ Publickey<Scheme::CKKS>
        public_key(Secretkey<Scheme::CKKS>& sk,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Publickey<Scheme::CKKS> output(context_);
            generate_public_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Relinkey of sk; see generate_relin_key.
         */
        __host__ Relinkey<Scheme::CKKS>
        relin_key(Secretkey<Scheme::CKKS>& sk,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            Relinkey<Scheme::CKKS> output(context_);
            generate_relin_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Galoiskey of sk; see generate_galois_key.
         */
        __host__ Galoiskey<Scheme::CKKS>
        galois_key(Secretkey<Scheme::CKKS>& sk,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Galoiskey<Scheme::CKKS> output(context_);
            generate_galois_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns a new Galoiskey of sk for the given shifts; see
         * generate_galois_key.
         */
        __host__ Galoiskey<Scheme::CKKS>
        galois_key(Secretkey<Scheme::CKKS>& sk, std::vector<int>& shifts,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            Galoiskey<Scheme::CKKS> output(context_, shifts);
            generate_galois_key(output, sk, options);
            return output;
        }

        /**
         * @brief Returns the keys CKKS bootstrapping uses: a Relinkey of sk,
         * a Galoiskey of sk for galois_shifts (the bootstrapping_key_indices()
         * of the operator) and, unless sparse_secret_key_hamming_weight is 0,
         * the switch keys between sk and a new sparse secret key with that
         * Hamming weight.
         */
        __host__ CKKSBootstrappingKeys bootstrapping_keys(
            Secretkey<Scheme::CKKS>& sk, std::vector<int> galois_shifts,
            int sparse_secret_key_hamming_weight = 0,
            const ExecutionOptions& options = ExecutionOptions())
        {
            CKKSBootstrappingKeys keys{relin_key(sk, options),
                                       galois_key(sk, galois_shifts, options),
                                       std::nullopt, std::nullopt};
            if (sparse_secret_key_hamming_weight > 0)
            {
                Secretkey<Scheme::CKKS> sparse_sk(
                    context_, sparse_secret_key_hamming_weight);
                generate_secret_key(sparse_sk, options);
                keys.dense_to_sparse.emplace(context_);
                generate_switch_key(*keys.dense_to_sparse, sparse_sk, sk,
                                    options);
                keys.sparse_to_dense.emplace(context_);
                generate_switch_key(*keys.sparse_to_dense, sk, sparse_sk,
                                    options);
            }
            return keys;
        }

      private:
        __host__ void generate_ckks_relin_key(Relinkey<Scheme::CKKS>& rk,
                                              Secretkey<Scheme::CKKS>& sk,
                                              const ExecutionOptions& options);

        __host__ void generate_ckks_galois_key(Galoiskey<Scheme::CKKS>& gk,
                                               Secretkey<Scheme::CKKS>& sk,
                                               const ExecutionOptions& options);

        __host__ void generate_ckks_switch_key(Switchkey<Scheme::CKKS>& swk,
                                               Secretkey<Scheme::CKKS>& new_sk,
                                               Secretkey<Scheme::CKKS>& old_sk,
                                               const ExecutionOptions& options);

      private:
        HEContext<Scheme::CKKS> context_;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        RNGSeed new_seed_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_KEYGENERATOR_H
