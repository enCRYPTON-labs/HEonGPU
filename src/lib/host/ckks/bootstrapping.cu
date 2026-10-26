// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/bootstrapping.cuh>
#include <heongpu/host/ckks/operator.cuh>

namespace heongpu
{
    namespace
    {
        // EvalMod evaluates a Chebyshev polynomial of the given degree and
        // then applies the double angle formula.
        int eval_mod_depth(const EvalModConfig& config, int degree)
        {
            return HEArithmeticOperator<Scheme::CKKS>::polynomial_depth(
                       degree) +
                   config.double_angle_;
        }

        BootstrappingLevels
        levels_with_slot_to_coeff_first(const BootstrappingConfig& config,
                                        int eval_mod)
        {
            BootstrappingLevels levels;
            levels.coeff_to_slot = config.CtoS_piece_;
            levels.eval_mod = eval_mod;
            levels.slot_to_coeff = config.StoC_piece_;
            levels.input_level = config.stc_config_.level_start_;
            levels.output_level = config.cts_config_.level_start_ -
                                  levels.coeff_to_slot - levels.eval_mod;
            return levels;
        }
    } // namespace

    BootstrappingLevels bootstrapping_levels(const BootstrappingConfig& config,
                                             arithmetic_bootstrapping_type type)
    {
        int eval_mod = eval_mod_depth(config.eval_mod_config_,
                                      config.eval_mod_config_.sine_deg_);
        switch (type)
        {
            case arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING:
            {
                BootstrappingLevels levels;
                levels.coeff_to_slot = config.CtoS_piece_;
                levels.eval_mod = eval_mod;
                levels.slot_to_coeff = config.StoC_piece_;
                levels.input_level = 0;
                levels.output_level =
                    config.cts_config_.level_start_ - levels.depth();
                return levels;
            }
            case arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING:
                return levels_with_slot_to_coeff_first(config, eval_mod);
            default:
                throw ParameterError("Invalid arithmetic bootstrapping type!");
        }
    }

    BootstrappingLevels bootstrapping_levels(const BootstrappingConfig& config,
                                             logic_bootstrapping_type type)
    {
        // The cosine of HELogicOperator::generate_bootstrapping_params has
        // the degree 2 * period * K - 2.
        int period;
        switch (type)
        {
            case logic_bootstrapping_type::BIT_BOOTSTRAPPING:
                period = 2;
                break;
            case logic_bootstrapping_type::GATE_BOOTSTRAPPING:
                period = 3;
                break;
            default:
                throw ParameterError("Invalid logic bootstrapping type!");
        }
        int degree = 2 * period * config.eval_mod_config_.K_ - 2;
        return levels_with_slot_to_coeff_first(
            config, eval_mod_depth(config.eval_mod_config_, degree));
    }

    namespace
    {
        // The configuration of the examples: SlotToCoeff from stc_start,
        // EvalMod from level 20 and CoeffToSlot from level 24.
        BootstrappingConfig example_config(int stc_start,
                                           double message_ratio = 256.0)
        {
            EvalModConfig eval_mod_config(20);
            eval_mod_config.message_ratio_ = message_ratio;
            return BootstrappingConfig(
                EncodingMatrixConfig(LinearTransformType::SLOTS_TO_COEFFS,
                                     stc_start),
                eval_mod_config,
                EncodingMatrixConfig(LinearTransformType::COEFFS_TO_SLOTS, 24));
        }

        CKKSBootstrappingPreset example_preset(std::vector<int> q_bit_sizes)
        {
            CKKSBootstrappingPreset preset;
            preset.poly_modulus_degree = 1 << 16;
            preset.coeff_modulus_bit_sizes = std::move(q_bit_sizes);
            preset.special_modulus_bit_sizes = {60, 60, 60, 60, 60};
            preset.scale = std::pow(2.0, 40);
            preset.secret_key_hamming_weight = 192;
            preset.sparse_secret_key_hamming_weight = 32;
            return preset;
        }

        // Q0, n primes of 40 bits, m primes of 60 bits and 4 primes of 56
        // bits for CoeffToSlot.
        std::vector<int> example_bit_sizes(int n, int m)
        {
            std::vector<int> sizes = {60};
            sizes.insert(sizes.end(), n, 40);
            sizes.insert(sizes.end(), m, 60);
            sizes.insert(sizes.end(), 4, 56);
            return sizes;
        }
    } // namespace

    CKKSBootstrappingPreset CKKSBootstrappingPreset::regular()
    {
        CKKSBootstrappingPreset preset = example_preset({});
        preset.arithmetic_type =
            arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING;
        // Q0 | 9 levels | 3 SlotToCoeff | 8 EvalMod | 4 CoeffToSlot
        preset.coeff_modulus = {
            0x10000000006e0001, 0x10000140001,      0xffffe80001,
            0xffffc40001,       0x100003e0001,      0xffffb20001,
            0x10000500001,      0xffff940001,       0xffff8a0001,
            0xffff820001,       0x7fffe60001,       0x7fffe40001,
            0x7fffe00001,       0xfffffffff840001,  0x1000000000860001,
            0xfffffffff6a0001,  0x1000000000980001, 0xfffffffff5a0001,
            0x1000000000b00001, 0x1000000000ce0001, 0xfffffffff2a0001,
            0x100000000060001,  0xfffffffff00001,   0xffffffffd80001,
            0x1000000002a0001};
        preset.special_modulus = {0x1fffffffffe00001, 0x1fffffffffc80001,
                                  0x1fffffffffb40001, 0x1fffffffff500001,
                                  0x1fffffffff420001};
        preset.coeff_modulus_bit_sizes.clear();
        preset.special_modulus_bit_sizes.clear();
        preset.config = example_config(12);
        return preset;
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::slim()
    {
        // Q0 | 3 SlotToCoeff + 9 levels | 8 EvalMod | 4 CoeffToSlot
        CKKSBootstrappingPreset preset =
            example_preset(example_bit_sizes(12, 8));
        preset.arithmetic_type =
            arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING;
        // SlotToCoeff runs first, so slot values become coefficients
        // directly; a larger message ratio keeps sin(x) ~ x accurate.
        preset.config = example_config(3, 1024.0);
        return preset;
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::bit()
    {
        // Q0 | 3 SlotToCoeff + 8 levels | 9 EvalMod | 4 CoeffToSlot
        CKKSBootstrappingPreset preset =
            example_preset(example_bit_sizes(11, 9));
        preset.logic_type = logic_bootstrapping_type::BIT_BOOTSTRAPPING;
        preset.config = example_config(3);
        return preset;
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::gate()
    {
        // Q0 | 3 SlotToCoeff + 7 levels | 10 EvalMod | 4 CoeffToSlot
        CKKSBootstrappingPreset preset =
            example_preset(example_bit_sizes(10, 10));
        preset.logic_type = logic_bootstrapping_type::GATE_BOOTSTRAPPING;
        preset.config = example_config(3);
        return preset;
    }

    void
    CKKSBootstrappingPreset::apply(const HEContext<Scheme::CKKS>& context) const
    {
        context->set_poly_modulus_degree(poly_modulus_degree);
        if (!coeff_modulus.empty())
        {
            context->set_coeff_modulus_values(coeff_modulus, special_modulus);
        }
        else
        {
            context->set_coeff_modulus_bit_sizes(coeff_modulus_bit_sizes,
                                                 special_modulus_bit_sizes);
        }
    }

    BootstrappingLevels CKKSBootstrappingPreset::levels() const
    {
        if (arithmetic_type != arithmetic_bootstrapping_type::NONE)
        {
            return bootstrapping_levels(config, arithmetic_type);
        }
        return bootstrapping_levels(config, logic_type);
    }

} // namespace heongpu
