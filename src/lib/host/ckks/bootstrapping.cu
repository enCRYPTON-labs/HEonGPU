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
            case arithmetic_bootstrapping_type::regular:
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
            case arithmetic_bootstrapping_type::slim:
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
            case logic_bootstrapping_type::bit:
                period = 2;
                break;
            case logic_bootstrapping_type::gate:
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
        // Levels of SlotToCoeff and CoeffToSlot in the presets.
        constexpr int slot_to_coeff_levels = 3;
        constexpr int coeff_to_slot_levels = 4;

        // The configuration of the examples: CoeffToSlot from the top level,
        // EvalMod right below it and SlotToCoeff from stc_start.
        BootstrappingConfig example_config(int stc_start, int top_level,
                                           double message_ratio = 256.0)
        {
            EvalModConfig eval_mod_config(top_level - coeff_to_slot_levels);
            eval_mod_config.message_ratio_ = message_ratio;
            return BootstrappingConfig(
                EncodingMatrixConfig(LinearTransformType::SLOTS_TO_COEFFS,
                                     stc_start),
                eval_mod_config,
                EncodingMatrixConfig(LinearTransformType::COEFFS_TO_SLOTS,
                                     top_level));
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
            sizes.insert(sizes.end(), coeff_to_slot_levels, 56);
            return sizes;
        }

        // The presets that run SlotToCoeff first: Q0 | 3 SlotToCoeff +
        // levels | eval_mod EvalMod | 4 CoeffToSlot.
        CKKSBootstrappingPreset slot_to_coeff_first(int levels, int eval_mod,
                                                    double message_ratio)
        {
            CKKSBootstrappingPreset preset = example_preset(
                example_bit_sizes(slot_to_coeff_levels + levels, eval_mod));
            preset.config = example_config(slot_to_coeff_levels,
                                           slot_to_coeff_levels + levels +
                                               eval_mod + coeff_to_slot_levels,
                                           message_ratio);
            return preset;
        }

        // Q0 | levels | 3 SlotToCoeff | 8 EvalMod | 4 CoeffToSlot. The
        // first levels user primes are the ones of the example, followed by
        // primes around 2^40.
        CKKSBootstrappingPreset regular_preset(int levels)
        {
            const std::vector<Data64> q0 = {0x10000000006e0001};
            const std::vector<Data64> user = {
                0x10000140001, 0xffffe80001, 0xffffc40001,
                0x100003e0001, 0xffffb20001, 0x10000500001,
                0xffff940001,  0xffff8a0001, 0xffff820001};
            const std::vector<Data64> bootstrapping = {
                0x7fffe60001,       0x7fffe40001,       0x7fffe00001,
                0xfffffffff840001,  0x1000000000860001, 0xfffffffff6a0001,
                0x1000000000980001, 0xfffffffff5a0001,  0x1000000000b00001,
                0x1000000000ce0001, 0xfffffffff2a0001,  0x100000000060001,
                0xfffffffff00001,   0xffffffffd80001,   0x1000000002a0001};

            CKKSBootstrappingPreset preset = example_preset({});
            preset.arithmetic_type = arithmetic_bootstrapping_type::regular;
            preset.coeff_modulus_bit_sizes.clear();
            preset.special_modulus_bit_sizes.clear();
            preset.special_modulus = {0x1fffffffffe00001, 0x1fffffffffc80001,
                                      0x1fffffffffb40001, 0x1fffffffff500001,
                                      0x1fffffffff420001};

            std::vector<Data64> levels_primes(
                user.begin(),
                user.begin() + std::min<std::size_t>(levels, user.size()));
            if (levels > static_cast<int>(user.size()))
            {
                std::vector<Data64> excluded;
                excluded.reserve(q0.size() + user.size() +
                                 bootstrapping.size());
                for (const std::vector<Data64>* primes :
                     {&q0, &user, &bootstrapping})
                {
                    excluded.insert(excluded.end(), primes->begin(),
                                    primes->end());
                }
                std::vector<Data64> more =
                    generate_primes_around(preset.poly_modulus_degree, 40,
                                           levels - user.size(), excluded);
                levels_primes.insert(levels_primes.end(), more.begin(),
                                     more.end());
            }

            preset.coeff_modulus = q0;
            preset.coeff_modulus.insert(preset.coeff_modulus.end(),
                                        levels_primes.begin(),
                                        levels_primes.end());
            preset.coeff_modulus.insert(preset.coeff_modulus.end(),
                                        bootstrapping.begin(),
                                        bootstrapping.end());
            preset.config = example_config(
                levels + slot_to_coeff_levels,
                static_cast<int>(preset.coeff_modulus.size()) - 1);
            return preset;
        }

        CKKSBootstrappingPreset slim_preset(int levels)
        {
            // SlotToCoeff runs first, so slot values become coefficients
            // directly; a larger message ratio keeps sin(x) ~ x accurate.
            CKKSBootstrappingPreset preset =
                slot_to_coeff_first(levels, 8, 1024.0);
            preset.arithmetic_type = arithmetic_bootstrapping_type::slim;
            return preset;
        }

        CKKSBootstrappingPreset bit_preset(int levels)
        {
            CKKSBootstrappingPreset preset =
                slot_to_coeff_first(levels, 9, 256.0);
            preset.logic_type = logic_bootstrapping_type::bit;
            return preset;
        }

        CKKSBootstrappingPreset gate_preset(int levels)
        {
            CKKSBootstrappingPreset preset =
                slot_to_coeff_first(levels, 10, 256.0);
            preset.logic_type = logic_bootstrapping_type::gate;
            return preset;
        }

        int bit_count(const std::vector<Data64>& primes,
                      const std::vector<int>& bit_sizes)
        {
            int count = 0;
            for (Data64 prime : primes)
            {
                count += calculate_bit_size(prime);
            }
            for (int size : bit_sizes)
            {
                count += size;
            }
            return count;
        }

        int total_bit_count(const CKKSBootstrappingPreset& preset)
        {
            return bit_count(preset.coeff_modulus,
                             preset.coeff_modulus_bit_sizes) +
                   bit_count(preset.special_modulus,
                             preset.special_modulus_bit_sizes);
        }

        // Returns build(levels), or throws ParameterError if levels is not
        // positive or the moduli exceed the 128-bit security limit.
        CKKSBootstrappingPreset
        with_levels(CKKSBootstrappingPreset (*build)(int), const char* name,
                    int levels)
        {
            if (levels < 1)
            {
                throw ParameterError(
                    error_message("Invalid number of usable levels ", levels,
                                  " for ", name, "! Expected 1 or more."));
            }
            CKKSBootstrappingPreset preset = build(levels);
            int limit = max_coeff_bit_count(sec_level_type::sec128,
                                            preset.poly_modulus_degree);
            int needed = total_bit_count(preset);
            if (needed > limit)
            {
                int fitting = levels - 1;
                while ((fitting > 0) &&
                       (total_bit_count(build(fitting)) > limit))
                {
                    fitting--;
                }
                throw ParameterError(error_message(
                    "The ", name, " bootstrapping preset with ", levels,
                    " usable levels does not fit sec128 with N = ",
                    preset.poly_modulus_degree, ": Q and P need ", needed,
                    " bits, the limit is ", limit, ". At most ", fitting,
                    " usable levels fit."));
            }
            return preset;
        }
    } // namespace

    CKKSBootstrappingPreset CKKSBootstrappingPreset::regular()
    {
        return regular(9);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::regular(int usable_levels)
    {
        return with_levels(regular_preset, "regular", usable_levels);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::slim()
    {
        return slim(9);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::slim(int usable_levels)
    {
        return with_levels(slim_preset, "slim", usable_levels);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::bit()
    {
        return bit(8);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::bit(int usable_levels)
    {
        return with_levels(bit_preset, "bit", usable_levels);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::gate()
    {
        return gate(7);
    }

    CKKSBootstrappingPreset CKKSBootstrappingPreset::gate(int usable_levels)
    {
        return with_levels(gate_preset, "gate", usable_levels);
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
        if (arithmetic_type != arithmetic_bootstrapping_type::none)
        {
            return bootstrapping_levels(config, arithmetic_type);
        }
        return bootstrapping_levels(config, logic_type);
    }

} // namespace heongpu
