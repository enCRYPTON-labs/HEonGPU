// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

// The bootstrapping presets, the levels computed for them and the key
// bundle, with inputs at the top level that bootstrapping brings down itself.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Preset = heongpu::CKKSBootstrappingPreset;

    struct Environment
    {
        explicit Environment(const Preset& preset)
        {
            context = heongpu::GenHEContext<S>(heongpu::sec_level_type::sec128);
            preset.apply(context);
            context->generate();
            generator = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret_key = std::make_unique<heongpu::Secretkey<S>>(
                context, preset.secret_key_hamming_weight);
            generator->generate_secret_key(*secret_key);
            auto public_key = generator->public_key(*secret_key);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret_key);
        }

        heongpu::CKKSBootstrappingKeys keys(const Preset& preset,
                                            std::vector<int> indices)
        {
            return generator->bootstrapping_keys(
                *secret_key, indices, preset.sparse_secret_key_hamming_weight);
        }

        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> generator;
        std::unique_ptr<heongpu::Secretkey<S>> secret_key;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
    };

    double max_error(const std::vector<double>& expected,
                     const std::vector<double>& values)
    {
        double error = 0.0;
        for (size_t i = 0; i < expected.size(); i++)
        {
            error = std::max(error, std::abs(expected[i] - values[i]));
        }
        return error;
    }

    // Bootstraps with preset, which must be a regular or slim preset.
    void check_arithmetic(const Preset& preset)
    {
        Environment setup(preset);
        heongpu::HEArithmeticOperator<S> operators(setup.context,
                                                   *setup.encoder);
        operators.generate_bootstrapping_params(preset);
        auto keys = setup.keys(preset, operators.bootstrapping_key_indices());
        ASSERT_TRUE(keys.dense_to_sparse.has_value());
        ASSERT_TRUE(keys.sparse_to_dense.has_value());

        const int slots = static_cast<int>(preset.poly_modulus_degree / 2);
        std::vector<double> message(slots);
        for (int i = 0; i < slots; i++)
        {
            message[i] = 0.5 * std::sin(0.01 * i);
        }
        auto plaintext = setup.encoder->encode(message, preset.scale);
        auto ciphertext = setup.encryptor->encrypt(plaintext);

        // The input is at the top level; bootstrapping brings it down.
        bool regular = preset.arithmetic_type ==
                       heongpu::arithmetic_bootstrapping_type::regular;
        auto result = regular
                          ? operators.regular_bootstrapping(ciphertext, keys)
                          : operators.slim_bootstrapping(ciphertext, keys);
        EXPECT_EQ(result.level(), preset.levels().output_level);

        auto decrypted = setup.decryptor->decrypt(result);
        auto values = setup.encoder->decode(decrypted);
        // The examples reach at least 15 bits (regular) and 22 bits (slim).
        EXPECT_LT(max_error(message, values),
                  regular ? std::pow(2.0, -12) : std::pow(2.0, -18));

        if (!regular)
        {
            // An input below the SlotToCoeff start level is an error.
            auto low = ciphertext;
            while (low.level() >= preset.levels().input_level)
            {
                operators.mod_drop_inplace(low);
            }
            EXPECT_THROW(operators.slim_bootstrapping(low, keys),
                         heongpu::LevelMismatchError);
        }

        EXPECT_THROW(operators.generate_bootstrapping_params(Preset::bit()),
                     heongpu::ParameterError);
    }

    // Bootstraps with preset, which must be a bit or gate preset.
    void check_logic(const Preset& preset)
    {
        Environment setup(preset);
        heongpu::HELogicOperator<S> operators(setup.context, *setup.encoder,
                                              preset.scale);
        operators.generate_bootstrapping_params(preset);
        auto keys = setup.keys(preset, operators.bootstrapping_key_indices());

        const int slots = static_cast<int>(preset.poly_modulus_degree / 2);
        std::vector<double> a(slots), b(slots);
        for (int i = 0; i < slots; i++)
        {
            a[i] = i % 2;
            b[i] = (i / 2) % 2;
        }
        auto plaintext_a = setup.encoder->encode(a, preset.scale);
        auto plaintext_b = setup.encoder->encode(b, preset.scale);
        auto x = setup.encryptor->encrypt(plaintext_a);
        auto y = setup.encryptor->encrypt(plaintext_b);

        bool bit = preset.logic_type == heongpu::logic_bootstrapping_type::bit;
        std::vector<double> expected(slots);
        for (int i = 0; i < slots; i++)
        {
            expected[i] =
                bit ? a[i] : static_cast<double>(int(a[i]) ^ int(b[i]));
        }

        // The inputs are at the top level; bootstrapping brings them down.
        auto result = bit ? operators.bit_bootstrapping(x, keys)
                          : operators.XOR_bootstrapping(x, y, keys);
        EXPECT_EQ(result.level(), preset.levels().output_level);

        auto decrypted = setup.decryptor->decrypt(result);
        auto values = setup.encoder->decode(decrypted);
        EXPECT_LT(max_error(expected, values), std::pow(2.0, -16));

        EXPECT_THROW(operators.generate_bootstrapping_params(Preset::regular()),
                     heongpu::ParameterError);
    }
} // namespace

TEST(BootstrappingPresets, Levels)
{
    struct Expected
    {
        Preset preset;
        int eval_mod, input_level, output_level;
    };
    std::vector<Expected> cases = {{Preset::regular(), 8, 0, 9},
                                   {Preset::slim(), 8, 3, 12},
                                   {Preset::bit(), 9, 3, 11},
                                   {Preset::gate(), 10, 3, 10}};
    for (const auto& c : cases)
    {
        heongpu::BootstrappingLevels levels = c.preset.levels();
        EXPECT_EQ(levels.coeff_to_slot, 4);
        EXPECT_EQ(levels.eval_mod, c.eval_mod);
        EXPECT_EQ(levels.slot_to_coeff, 3);
        EXPECT_EQ(levels.depth(), 7 + c.eval_mod);
        EXPECT_EQ(levels.input_level, c.input_level);
        EXPECT_EQ(levels.output_level, c.output_level);
    }
    EXPECT_EQ(Preset::regular().levels().usable_levels(), 9);
    EXPECT_EQ(Preset::slim().levels().usable_levels(), 9);
    EXPECT_EQ(Preset::bit().levels().usable_levels(), 8);
    EXPECT_EQ(Preset::gate().levels().usable_levels(), 7);
}

TEST(BootstrappingPresets, Usable_Levels)
{
    // The examples: CoeffToSlot from level 24 and EvalMod from level 20.
    for (const Preset& preset :
         {Preset::regular(), Preset::slim(), Preset::bit(), Preset::gate()})
    {
        EXPECT_EQ(preset.config.cts_config_.level_start_, 24);
        EXPECT_EQ(preset.config.eval_mod_config_.level_start_, 20);
    }
    EXPECT_EQ(Preset::regular().config.stc_config_.level_start_, 12);
    EXPECT_EQ(Preset::slim().config.stc_config_.level_start_, 3);
    EXPECT_EQ(Preset::regular().coeff_modulus.size(), 25u);
    EXPECT_EQ(Preset::regular().coeff_modulus[9], 0xffff820001);

    struct Case
    {
        Preset (*preset)(int);
        int max_levels;
    };
    for (const Case& c : {Case{Preset::regular, 14}, Case{Preset::slim, 14},
                          Case{Preset::bit, 12}, Case{Preset::gate, 11}})
    {
        for (int levels = 1; levels <= c.max_levels; levels++)
        {
            Preset preset = c.preset(levels);
            heongpu::BootstrappingLevels used = preset.levels();
            EXPECT_EQ(used.usable_levels(), levels);
            EXPECT_EQ(used.depth(), 7 + used.eval_mod);
            int q_count = static_cast<int>(
                std::max(preset.coeff_modulus.size(),
                         preset.coeff_modulus_bit_sizes.size()));
            EXPECT_EQ(preset.config.cts_config_.level_start_, q_count - 1);

            // The context accepts the moduli.
            auto context =
                heongpu::GenHEContext<S>(heongpu::sec_level_type::sec128);
            EXPECT_NO_THROW(preset.apply(context)) << levels;
        }
        EXPECT_THROW(c.preset(c.max_levels + 1), heongpu::ParameterError);
        EXPECT_THROW(c.preset(0), heongpu::ParameterError);
    }

    // The levels after the ones of regular() are primes around 2^40.
    Preset regular = Preset::regular(12);
    Preset example = Preset::regular();
    for (int i = 0; i <= 9; i++)
    {
        EXPECT_EQ(regular.coeff_modulus[i], example.coeff_modulus[i]);
    }
    for (int i = 10; i <= 12; i++)
    {
        EXPECT_NEAR(std::log2(double(regular.coeff_modulus[i])), 40.0, 1e-3);
    }

    try
    {
        Preset::regular(15);
    }
    catch (const heongpu::ParameterError& error)
    {
        EXPECT_STREQ(error.what(),
                     "The regular bootstrapping preset with 15 usable levels "
                     "does not fit sec128 with N = 65536: Q and P need 1799 "
                     "bits, the limit is 1761. At most 14 usable levels fit.");
    }
}

TEST(BootstrappingPresets, Regular_And_Slim)
{
    for (const Preset& preset : {Preset::regular(), Preset::slim()})
    {
        check_arithmetic(preset);
    }
}

// Each test runs in its own process, so these do not share the memory pool
// with the ones above.
TEST(BootstrappingPresets, Regular_And_Slim_With_Other_Levels)
{
    for (const Preset& preset : {Preset::regular(14), Preset::slim(3)})
    {
        check_arithmetic(preset);
    }
}

TEST(BootstrappingPresets, Bit_And_Gate)
{
    for (const Preset& preset : {Preset::bit(), Preset::gate()})
    {
        check_logic(preset);
    }
}

TEST(BootstrappingPresets, Bit_And_Gate_With_Other_Levels)
{
    for (const Preset& preset : {Preset::bit(12), Preset::gate(2)})
    {
        check_logic(preset);
    }
}

TEST(BootstrappingPresets, Keys_Without_A_Sparse_Secret_Key)
{
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();

    auto keys = generator.bootstrapping_keys(secret_key, {1, 2, 4});
    EXPECT_FALSE(keys.dense_to_sparse.has_value());
    EXPECT_FALSE(keys.sparse_to_dense.has_value());

    auto sparse_keys = generator.bootstrapping_keys(secret_key, {1, 2, 4}, 16);
    EXPECT_TRUE(sparse_keys.dense_to_sparse.has_value());
    EXPECT_TRUE(sparse_keys.sparse_to_dense.has_value());
}
