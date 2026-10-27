// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// The rescale modes of the CKKS context and the scales of the levels in the
// automatic mode.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Mode = heongpu::rescale_mode;

    heongpu::HEParams<S> params(int depth, Mode mode = Mode::automatic)
    {
        heongpu::HEParams<S> p;
        p.multiplicative_depth = depth;
        p.rescale_mode = mode;
        return p;
    }
} // namespace

TEST(RescaleMode, Manual_By_Default)
{
    auto context = params(3, Mode::manual).make_context();
    EXPECT_EQ(context->get_rescale_mode(), Mode::manual);
    EXPECT_THROW(context->get_level_scale(0), heongpu::StateError);
    context->generate();
    EXPECT_EQ(context->to_string().find("rescale"), std::string::npos);
}

TEST(RescaleMode, Needs_The_Moduli_And_A_Scale)
{
    auto empty = heongpu::GenHEContext<S>(heongpu::sec_level_type::sec128);
    EXPECT_THROW(empty->set_rescale_mode(Mode::automatic, std::pow(2.0, 40)),
                 heongpu::StateError);

    auto context = params(3, Mode::manual).make_context();
    for (double scale : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(context->set_rescale_mode(Mode::automatic, scale),
                     heongpu::ParameterError);
    }
    EXPECT_EQ(context->get_rescale_mode(), Mode::manual);
    EXPECT_THROW(context->set_rescale_mode(static_cast<Mode>(7)),
                 heongpu::ParameterError);
}

TEST(RescaleMode, Scales_Of_The_Levels)
{
    auto set = params(10).generate();
    EXPECT_EQ(set.rescale_mode, Mode::automatic);
    EXPECT_NE(set.to_string().find(", rescale=automatic)"), std::string::npos);

    auto context = set.make_context();
    ASSERT_EQ(context->get_rescale_mode(), Mode::automatic);
    EXPECT_EQ(context->get_level_scale(10), std::pow(2.0, 40));
    for (int level = 10; level > 0; level--)
    {
        double scale = context->get_level_scale(level);
        EXPECT_EQ(context->get_level_scale(level - 1),
                  scale * scale / double(set.coeff_modulus[level]));
        // HEParams keeps them next to the scale.
        EXPECT_NEAR(std::log2(context->get_level_scale(level - 1)), 40.0, 1e-4);
    }
    EXPECT_THROW(context->get_level_scale(-1), heongpu::ParameterError);
    EXPECT_THROW(context->get_level_scale(11), heongpu::ParameterError);

    // Back to manual.
    context->set_rescale_mode(Mode::manual);
    EXPECT_THROW(context->get_level_scale(10), heongpu::StateError);
}

TEST(RescaleMode, Not_Part_Of_The_Identity)
{
    auto manual = params(3, Mode::manual).make_context();
    auto automatic = params(3).make_context();
    manual->generate();
    automatic->generate();
    EXPECT_EQ(manual->context_id(), automatic->context_id());
    EXPECT_NE(automatic->to_string().find(", rescale=automatic, id="),
              std::string::npos);
}

TEST(RescaleMode, Products_Have_The_Scales_Of_The_Levels)
{
    // Multiplying two ciphertexts of a level and rescaling gives exactly
    // the scale of the next level.
    constexpr int depth = 6;
    auto context = params(depth).make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> generator(context);
    auto secret_key = generator.secret_key();
    auto public_key = generator.public_key(secret_key);
    auto relin_key = generator.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> x(context->get_poly_modulus_degree() / 2, 0.5);
    auto plaintext = encoder.encode(x, context->get_level_scale(depth));
    auto ciphertext = encryptor.encrypt(plaintext);
    for (int level = depth; level > 0; level--)
    {
        operators.multiply_inplace(ciphertext, ciphertext);
        operators.relinearize_inplace(ciphertext, relin_key);
        operators.rescale_inplace(ciphertext);
        ASSERT_EQ(ciphertext.level(), level - 1);
        EXPECT_EQ(ciphertext.scale(), context->get_level_scale(level - 1));
    }
}

TEST(RescaleMode, Encoding_With_The_Scales_Of_The_Levels)
{
    constexpr int depth = 5;
    auto manual = params(depth, Mode::manual).make_context();
    manual->generate();
    heongpu::HEEncoder<S> manual_encoder(manual);
    std::vector<double> x(manual->get_poly_modulus_degree() / 2, 0.25);
    EXPECT_THROW(manual_encoder.encode(x), heongpu::StateError);
    EXPECT_THROW(manual_encoder.encode(x, 2), heongpu::StateError);

    auto context = params(depth).make_context();
    context->generate();
    heongpu::HEEncoder<S> encoder(context);

    auto top = encoder.encode(x);
    EXPECT_EQ(top.depth(), 0);
    EXPECT_EQ(top.scale(), context->get_level_scale(depth));
    for (int d = 0; d <= depth; d++)
    {
        double scale = context->get_level_scale(depth - d);
        auto plaintext = encoder.encode(x, d);
        EXPECT_EQ(plaintext.depth(), d);
        EXPECT_EQ(plaintext.scale(), scale);
        // The same plaintext as with the scale given.
        auto reference = encoder.encode(x, scale, d);
        EXPECT_EQ(encoder.decode(plaintext), encoder.decode(reference));
    }

    std::vector<Complex64> z(x.size(), Complex64(0.5, -0.25));
    EXPECT_EQ(encoder.encode(z, 3).scale(), context->get_level_scale(2));

    heongpu::HostVector<double> host(x.size(), 0.25);
    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, host, 1);
    EXPECT_EQ(plaintext.scale(), context->get_level_scale(depth - 1));

    EXPECT_THROW(encoder.encode(x, depth + 1), heongpu::ParameterError);
    EXPECT_THROW(encoder.encode(x, -1), heongpu::ParameterError);
}

namespace
{
    // Keys and operators for a context of depth 8 in one of the modes.
    struct Environment
    {
        explicit Environment(Mode mode)
        {
            context = params(depth, mode).make_context();
            context->generate();
            generator = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret_key = std::make_unique<heongpu::Secretkey<S>>(
                generator->secret_key());
            auto public_key = generator->public_key(*secret_key);
            relin_key = std::make_unique<heongpu::Relinkey<S>>(
                generator->relin_key(*secret_key));
            galois_key = std::make_unique<heongpu::Galoiskey<S>>(
                generator->galois_key(*secret_key));
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret_key);
            operators = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                context, *encoder);
            slots = context->get_poly_modulus_degree() / 2;
        }

        std::vector<double> values(double offset)
        {
            std::vector<double> x(slots);
            for (int i = 0; i < slots; i++)
            {
                x[i] = std::sin(0.001 * i + offset);
            }
            return x;
        }

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& x)
        {
            auto plaintext = encoder->encode(x, std::pow(2.0, 40));
            return encryptor->encrypt(plaintext);
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& ciphertext)
        {
            auto plaintext = decryptor->decrypt(ciphertext);
            return encoder->decode(plaintext);
        }

        static constexpr int depth = 8;
        int slots;
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> generator;
        std::unique_ptr<heongpu::Secretkey<S>> secret_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin_key;
        std::unique_ptr<heongpu::Galoiskey<S>> galois_key;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> operators;
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
} // namespace

TEST(RescaleMode, Manual_Code_Gives_The_Same_Results)
{
    // Multiplications, relinearizations and rescalings written for the manual
    // mode work the same in the automatic mode.
    std::vector<std::vector<double>> results;
    for (Mode mode : {Mode::manual, Mode::automatic})
    {
        Environment setup(mode);
        auto x = setup.encrypt(setup.values(0.0));
        auto y = setup.encrypt(setup.values(1.0));
        auto& ops = *setup.operators;
        for (int level = 0; level < 4; level++)
        {
            ops.multiply_inplace(x, y);
            ops.relinearize_inplace(x, *setup.relin_key);
            ops.rescale_inplace(x);
            ops.mod_drop_inplace(y);
        }
        EXPECT_EQ(x.level(), Environment::depth - 4);
        results.push_back(setup.decrypt(x));
    }
    EXPECT_LT(max_error(results[0], results[1]), 1e-5);
}

TEST(RescaleMode, Rescales_When_Needed)
{
    Environment setup(Mode::automatic);
    auto& ops = *setup.operators;
    auto xs = setup.values(0.0), ys = setup.values(1.0);
    auto x = setup.encrypt(xs);
    auto y = setup.encrypt(ys);

    // x^2 * y: the square is rescaled before the next multiplication, and y
    // is brought down to its level.
    auto square = ops.multiply(x, x);
    ops.relinearize_inplace(square, *setup.relin_key);
    EXPECT_TRUE(square.rescale_required());
    auto product = ops.multiply(square, y);
    ops.relinearize_inplace(product, *setup.relin_key);
    EXPECT_EQ(product.level(), Environment::depth - 1);
    EXPECT_TRUE(product.rescale_required());

    // The inputs are not changed.
    EXPECT_TRUE(square.rescale_required());
    EXPECT_EQ(square.level(), Environment::depth);
    EXPECT_EQ(y.level(), Environment::depth);
    EXPECT_EQ(y.scale(), setup.context->get_level_scale(Environment::depth));

    // A rotation rescales too.
    auto rotated = ops.rotate_rows(product, *setup.galois_key, 1);
    EXPECT_FALSE(rotated.rescale_required());
    EXPECT_EQ(rotated.level(), Environment::depth - 2);
    EXPECT_EQ(rotated.scale(),
              setup.context->get_level_scale(Environment::depth - 2));

    std::vector<double> expected(setup.slots), expected_rotated(setup.slots);
    for (int i = 0; i < setup.slots; i++)
    {
        expected[i] = xs[i] * xs[i] * ys[i];
    }
    for (int i = 0; i < setup.slots; i++)
    {
        expected_rotated[i] = expected[(i + 1) % setup.slots];
    }
    EXPECT_LT(max_error(expected, setup.decrypt(product)), 1e-6);
    EXPECT_LT(max_error(expected_rotated, setup.decrypt(rotated)), 1e-6);
}

TEST(RescaleMode, Adds_Ciphertexts_Of_Different_Levels)
{
    Environment setup(Mode::automatic);
    auto& ops = *setup.operators;
    auto xs = setup.values(0.0), ys = setup.values(1.0);
    auto x = setup.encrypt(xs);
    auto y = setup.encrypt(ys);

    // x^4 + x^2 + y, each at another level; the sum of two products that
    // both need a rescaling is not rescaled.
    auto square = ops.multiply(x, x);
    ops.relinearize_inplace(square, *setup.relin_key);
    auto fourth = ops.multiply(square, square);
    ops.relinearize_inplace(fourth, *setup.relin_key);
    auto both = ops.add(square, square);
    EXPECT_TRUE(both.rescale_required());
    auto sum = ops.add(fourth, square);
    sum = ops.add(sum, y);
    EXPECT_FALSE(sum.rescale_required());
    EXPECT_EQ(sum.level(), Environment::depth - 2);
    EXPECT_EQ(sum.scale(),
              setup.context->get_level_scale(Environment::depth - 2));

    // mod_drop keeps the scale of the level.
    auto dropped = ops.mod_drop(y);
    EXPECT_EQ(dropped.scale(),
              setup.context->get_level_scale(Environment::depth - 1));
    auto difference = ops.sub(dropped, y);

    std::vector<double> expected(setup.slots), zero(setup.slots, 0.0);
    for (int i = 0; i < setup.slots; i++)
    {
        double x2 = xs[i] * xs[i];
        expected[i] = x2 * x2 + x2 + ys[i];
    }
    EXPECT_LT(max_error(expected, setup.decrypt(sum)), 1e-6);
    EXPECT_LT(max_error(zero, setup.decrypt(difference)), 1e-6);
}

TEST(RescaleMode, Plaintexts_And_Constants)
{
    Environment setup(Mode::automatic);
    auto& ops = *setup.operators;
    auto xs = setup.values(0.0), ws = setup.values(2.0);
    auto x = setup.encrypt(xs);
    auto square = ops.multiply(x, x);
    ops.relinearize_inplace(square, *setup.relin_key);

    // The plaintext is encoded at the depth of the rescaled square.
    auto w = setup.encoder->encode(ws, 1);
    auto weighted = ops.multiply_plain(square, w);
    auto halved = ops.multiply_plain(weighted, 0.5, 0.0);
    // halved is rescaled to depth 3 for the addition.
    auto w3 = setup.encoder->encode(ws, 3);
    auto result = ops.add_plain(halved, w3);
    ops.add_plain_inplace(result, 0.25);
    EXPECT_EQ(result.depth(), 3);

    std::vector<double> expected(setup.slots);
    for (int i = 0; i < setup.slots; i++)
    {
        expected[i] = 0.5 * xs[i] * xs[i] * ws[i] + ws[i] + 0.25;
    }
    EXPECT_LT(max_error(expected, setup.decrypt(result)), 1e-6);

    // A plaintext of depth 1 brought down to depth 3 keeps the scale of
    // depth 1.
    EXPECT_THROW(ops.add_plain(halved, w), heongpu::ParameterError);

    // A plaintext at the depth of the ciphertext but of another scale.
    auto other = setup.encoder->encode(ws, 1.5 * std::pow(2.0, 40), 1);
    EXPECT_THROW(ops.multiply_plain(square, other), heongpu::ParameterError);
}

TEST(RescaleMode, Errors)
{
    Environment setup(Mode::automatic);
    auto& ops = *setup.operators;
    auto x = setup.encrypt(setup.values(0.0));

    // A ciphertext of another scale at the same level.
    auto plaintext =
        setup.encoder->encode(setup.values(1.0), 1.001 * std::pow(2.0, 40));
    auto other = setup.encryptor->encrypt(plaintext);
    EXPECT_THROW(ops.add(x, other), heongpu::ParameterError);

    // At another level its scale is corrected.
    auto square = ops.multiply(x, x);
    ops.relinearize_inplace(square, *setup.relin_key);
    EXPECT_NO_THROW(ops.add(square, other));

    // A rescaling needs a relinearization first.
    auto unrelinearized = ops.multiply(x, x);
    EXPECT_THROW(ops.multiply(unrelinearized, x), heongpu::StateError);

    // Bootstrapping and logic operations need the manual mode.
    EXPECT_THROW(ops.generate_bootstrapping_params(
                     heongpu::CKKSBootstrappingPreset::regular()),
                 heongpu::StateError);
    EXPECT_THROW(heongpu::HELogicOperator<S>(setup.context, *setup.encoder,
                                             std::pow(2.0, 40)),
                 heongpu::StateError);
}

TEST(RescaleMode, Polynomials)
{
    Environment setup(Mode::automatic);
    auto& ops = *setup.operators;
    auto xs = setup.values(0.0);
    auto x = setup.encrypt(xs);
    auto square = ops.multiply(x, x);
    ops.relinearize_inplace(square, *setup.relin_key);

    // 1 + t + t^2 / 2 + t^3 / 6 of t = x^2, which still needs a rescaling.
    std::vector<double> coefficients = {1.0, 1.0, 0.5, 1.0 / 6.0};
    auto result =
        ops.evaluate_polynomial(square, coefficients, *setup.relin_key);
    int level = Environment::depth - 1 -
                heongpu::HEArithmeticOperator<S>::polynomial_depth(3);
    EXPECT_EQ(result.level(), level);
    EXPECT_EQ(result.scale(), setup.context->get_level_scale(level));

    std::vector<double> expected(setup.slots);
    for (int i = 0; i < setup.slots; i++)
    {
        double t = xs[i] * xs[i];
        expected[i] = 1.0 + t + t * t / 2.0 + t * t * t / 6.0;
    }
    EXPECT_LT(max_error(expected, setup.decrypt(result)), 1e-5);

    // The result adds to ciphertexts of other levels.
    auto sum = ops.add(result, x);
    for (int i = 0; i < setup.slots; i++)
    {
        expected[i] += xs[i];
    }
    EXPECT_LT(max_error(expected, setup.decrypt(sum)), 1e-5);

    auto chebyshev = ops.evaluate_chebyshev(
        x, [](double v) { return std::exp(v); }, -2.0, 2.0, 7,
        *setup.relin_key);
    EXPECT_EQ(chebyshev.scale(),
              setup.context->get_level_scale(chebyshev.level()));
}
