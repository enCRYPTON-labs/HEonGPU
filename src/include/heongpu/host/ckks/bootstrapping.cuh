// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_BOOTSTRAPPING_H
#define HEONGPU_CKKS_BOOTSTRAPPING_H

#include <cstddef>
#include <vector>
#include <heongpu/util/util.cuh>
#include <heongpu/host/ckks/context.cuh>

namespace heongpu
{
    /**
     * @brief The levels a CKKS bootstrapping uses, see bootstrapping_levels.
     */
    struct BootstrappingLevels
    {
        int coeff_to_slot; // levels used by CoeffToSlot
        int eval_mod; // levels used by EvalMod
        int slot_to_coeff; // levels used by SlotToCoeff
        int input_level; // the level the input is brought to
        int output_level; // the level of the result

        // Levels used by the bootstrapping.
        int depth() const { return coeff_to_slot + eval_mod + slot_to_coeff; }

        // Levels left for the computation between two bootstrappings.
        int usable_levels() const { return output_level - input_level; }
    };

    /**
     * @brief Returns the levels a bootstrapping with config uses. Regular
     * bootstrapping takes its input at level 0 and runs SlotToCoeff last;
     * slim bootstrapping runs SlotToCoeff first, on its input at the
     * SlotToCoeff start level.
     */
    BootstrappingLevels
    bootstrapping_levels(const BootstrappingConfig& config,
                         arithmetic_bootstrapping_type type);

    /**
     * @brief Returns the levels a bit or gate bootstrapping with config uses.
     * Both run SlotToCoeff first, on their input at the SlotToCoeff start
     * level.
     */
    BootstrappingLevels bootstrapping_levels(const BootstrappingConfig& config,
                                             logic_bootstrapping_type type);

    /**
     * @brief Parameters for CKKS bootstrapping: the ring size and moduli of
     * the context, the scale, the Hamming weights of the secret keys and the
     * bootstrapping configuration.
     *
     * regular(), slim(), bit() and gate() return the parameters of the
     * bootstrapping examples, with N = 2^16 for 128-bit security:
     * @code
     * auto preset = heongpu::CKKSBootstrappingPreset::regular();
     * auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
     *     heongpu::sec_level_type::sec128);
     * preset.apply(context);
     * context->generate();
     * ...
     * operators.generate_bootstrapping_params(preset);
     * @endcode
     */
    struct CKKSBootstrappingPreset
    {
        static CKKSBootstrappingPreset regular();
        static CKKSBootstrappingPreset slim();
        static CKKSBootstrappingPreset bit();
        static CKKSBootstrappingPreset gate();

        // Exactly one of them is not NONE.
        arithmetic_bootstrapping_type arithmetic_type =
            arithmetic_bootstrapping_type::NONE;
        logic_bootstrapping_type logic_type = logic_bootstrapping_type::NONE;

        std::size_t poly_modulus_degree = 0;
        // The Q and P primes; when they are empty, primes of the bit sizes
        // below are generated.
        std::vector<Data64> coeff_modulus;
        std::vector<Data64> special_modulus;
        std::vector<int> coeff_modulus_bit_sizes;
        std::vector<int> special_modulus_bit_sizes;

        double scale = 0.0;
        int secret_key_hamming_weight = 0;
        // Hamming weight of the sparse secret key of HEKeyGenerator::
        // bootstrapping_keys; 0 for none.
        int sparse_secret_key_hamming_weight = 0;
        BootstrappingConfig config{EncodingMatrixConfig(), EvalModConfig(),
                                   EncodingMatrixConfig()};

        /**
         * @brief Sets the ring size and the moduli of context, before its
         * generate().
         */
        void apply(const HEContext<Scheme::CKKS>& context) const;

        /**
         * @brief Returns the levels the bootstrapping uses; see
         * bootstrapping_levels.
         */
        BootstrappingLevels levels() const;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_BOOTSTRAPPING_H
