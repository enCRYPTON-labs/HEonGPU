// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// evaluate_polynomial and evaluate_chebyshev against the same polynomials
// evaluated on the CPU, and the levels they use.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    constexpr int degree = 8192;
    constexpr int slots = degree / 2;

    double horner(const std::vector<double>& coefficients, double x)
    {
        double result = 0.0;
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
        {
            result = result * x + *it;
        }
        return result;
    }

    double clenshaw(const std::vector<double>& coefficients, double a, double b,
                    double x)
    {
        double y = (2.0 * x - a - b) / (b - a);
        double b1 = 0.0, b2 = 0.0;
        for (int k = static_cast<int>(coefficients.size()) - 1; k >= 1; k--)
        {
            double b0 = coefficients[k] + 2.0 * y * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        return coefficients[0] + y * b1 - b2;
    }

    class Polynomial : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes(
                {60, 40, 40, 40, 40, 40, 40, 40, 40}, {60});
            context->generate();

            heongpu::HEKeyGenerator<S> generator(context);
            secret_key =
                std::make_unique<heongpu::Secretkey<S>>(generator.secret_key());
            auto public_key = generator.public_key(*secret_key);
            relin_key = std::make_unique<heongpu::Relinkey<S>>(
                generator.relin_key(*secret_key));
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret_key);
            operators = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                context, *encoder);
        }

        std::vector<double> points(double a, double b)
        {
            std::vector<double> x(slots);
            for (int i = 0; i < slots; i++)
            {
                x[i] = a + (b - a) * static_cast<double>(i) / (slots - 1);
            }
            return x;
        }

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& x)
        {
            auto plaintext = encoder->encode(x, scale);
            return encryptor->encrypt(plaintext);
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& ciphertext)
        {
            auto plaintext = decryptor->decrypt(ciphertext);
            return encoder->decode(plaintext);
        }

        const double scale = std::pow(2.0, 40);
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::Secretkey<S>> secret_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin_key;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> operators;
    };
} // namespace

TEST_F(Polynomial, Monomial_Basis_Matches_CPU_And_Uses_Its_Depth)
{
    auto x = points(-1.0, 1.0);
    auto input = encrypt(x);

    for (int d = 1; d <= 17; d++)
    {
        std::vector<double> coefficients(d + 1);
        for (int i = 0; i <= d; i++)
        {
            coefficients[i] = std::cos(1.0 + 0.7 * i) / (i + 1);
        }

        auto result =
            operators->evaluate_polynomial(input, coefficients, *relin_key);
        EXPECT_EQ(result.level(),
                  input.level() -
                      heongpu::HEArithmeticOperator<S>::polynomial_depth(d))
            << "degree " << d;
        EXPECT_NEAR(result.scale() / input.scale(), 1.0, 1e-9)
            << "degree " << d;

        auto values = decrypt(result);
        for (int i = 0; i < slots; i++)
        {
            ASSERT_NEAR(values[i], horner(coefficients, x[i]), 1e-6)
                << "degree " << d << ", slot " << i;
        }
    }
}

TEST_F(Polynomial, Chebyshev_Series_On_Unit_Interval)
{
    auto x = points(-1.0, 1.0);
    auto input = encrypt(x);
    std::vector<double> coefficients = {0.1,   -0.4, 0.3, 0.2,
                                        -0.15, 0.05, 0.1, -0.02};

    auto result = operators->evaluate_chebyshev(input, coefficients, -1.0, 1.0,
                                                *relin_key);
    EXPECT_EQ(result.level(), input.level() - 3);
    EXPECT_NEAR(result.scale() / input.scale(), 1.0, 1e-9);

    auto values = decrypt(result);
    for (int i = 0; i < slots; i++)
    {
        ASSERT_NEAR(values[i], clenshaw(coefficients, -1.0, 1.0, x[i]), 1e-6)
            << "slot " << i;
    }
}

TEST_F(Polynomial, Sigmoid_And_Inverse)
{
    struct Case
    {
        const char* name;
        std::function<double(double)> function;
        double a, b;
        int degree;
    };
    std::vector<Case> cases = {
        {"sigmoid", [](double v) { return 1.0 / (1.0 + std::exp(-v)); }, -8.0,
         8.0, 31},
        {"inverse", [](double v) { return 1.0 / v; }, 1.0, 4.0, 31},
    };

    for (const auto& c : cases)
    {
        auto x = points(c.a, c.b);
        auto input = encrypt(x);
        auto coefficients =
            heongpu::chebyshev_coefficients(c.function, c.a, c.b, c.degree);

        auto result = operators->evaluate_chebyshev(input, c.function, c.a, c.b,
                                                    c.degree, *relin_key);
        EXPECT_EQ(result.level(),
                  input.level() -
                      heongpu::HEArithmeticOperator<S>::polynomial_depth(
                          c.degree, true, c.a, c.b))
            << c.name;
        EXPECT_NEAR(result.scale() / input.scale(), 1.0, 1e-9) << c.name;

        // The approximation error of the interpolant, measured on the CPU,
        // plus the error of the homomorphic evaluation.
        double approximation_error = 0.0;
        for (int i = 0; i < slots; i++)
        {
            approximation_error =
                std::max(approximation_error,
                         std::abs(clenshaw(coefficients, c.a, c.b, x[i]) -
                                  c.function(x[i])));
        }
        EXPECT_LT(approximation_error, 1e-4) << c.name;

        auto values = decrypt(result);
        for (int i = 0; i < slots; i++)
        {
            ASSERT_NEAR(values[i], clenshaw(coefficients, c.a, c.b, x[i]), 1e-6)
                << c.name << ", slot " << i;
            ASSERT_NEAR(values[i], c.function(x[i]), approximation_error + 1e-6)
                << c.name << ", slot " << i;
        }
    }
}

TEST_F(Polynomial, Invalid_Arguments)
{
    auto input = encrypt(points(-1.0, 1.0));
    std::vector<double> constant = {1.0};
    EXPECT_THROW(operators->evaluate_polynomial(input, constant, *relin_key),
                 heongpu::ParameterError);
    std::vector<double> linear = {0.0, 1.0};
    EXPECT_THROW(
        operators->evaluate_chebyshev(input, linear, 1.0, 1.0, *relin_key),
        heongpu::ParameterError);
    auto identity = [](double v) { return v; };
    EXPECT_THROW(operators->evaluate_chebyshev(input, identity, -1.0, 1.0, 0,
                                               *relin_key),
                 heongpu::ParameterError);

    // Degree 255 needs 8 levels, and 9 on [0, 1].
    std::vector<double> long_polynomial(256, 0.001);
    EXPECT_EQ(input.level(), 8);
    EXPECT_NO_THROW(
        operators->evaluate_polynomial(input, long_polynomial, *relin_key));
    EXPECT_THROW(operators->evaluate_chebyshev(input, long_polynomial, 0.0, 1.0,
                                               *relin_key),
                 heongpu::LevelMismatchError);
    operators->mod_drop_inplace(input);
    EXPECT_THROW(
        operators->evaluate_polynomial(input, long_polynomial, *relin_key),
        heongpu::LevelMismatchError);
}

TEST_F(Polynomial, User_Guide_Example)
{
    auto points_x = points(-1.0, 1.0);
    auto x = encrypt(points_x);
    auto& operators = *this->operators;
    auto& relin_key = *this->relin_key;

    // 0.5 + 0.25 x - 0.02 x^3, for slots x with |x| <= 1.
    auto p =
        operators.evaluate_polynomial(x, {0.5, 0.25, 0.0, -0.02}, relin_key);

    // The sigmoid on [-8, 8], approximated by a Chebyshev interpolant of
    // degree 31.
    auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    auto s = operators.evaluate_chebyshev(x, sigmoid, -8.0, 8.0, 31, relin_key);

    auto p_values = decrypt(p);
    auto s_values = decrypt(s);
    for (int i = 0; i < slots; i++)
    {
        double v = points_x[i];
        ASSERT_NEAR(p_values[i], 0.5 + 0.25 * v - 0.02 * v * v * v, 1e-6);
        ASSERT_NEAR(s_values[i], sigmoid(v), 1e-4);
    }
}
