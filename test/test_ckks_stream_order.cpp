// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

// Operations on a non-blocking stream give correct results. Their operands
// may be written on another stream: the copies that evaluate_polynomial and
// the automatic rescale mode make run on the stream of the operand, and the
// stream of the operation waits for them. The stream of the operands is held
// for a while on the host before they are written, so an operation that does
// not wait reads unwritten memory.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Cipher = heongpu::Ciphertext<S>;

    void hold(cudaStream_t stream)
    {
        ASSERT_EQ(cudaLaunchHostFunc(
                      stream,
                      [](void*) {
                          std::this_thread::sleep_for(
                              std::chrono::milliseconds(200));
                      },
                      nullptr),
                  cudaSuccess);
    }

    class CkksStreamOrder : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            heongpu::HEParams<S> params;
            params.multiplicative_depth = 4;
            params.rescale_mode = heongpu::rescale_mode::automatic;
            context = params.make_context();
            context->generate();
            slots = context->get_poly_modulus_degree() / 2;

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

            ASSERT_EQ(cudaStreamCreateWithFlags(&source, cudaStreamNonBlocking),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamCreateWithFlags(&target, cudaStreamNonBlocking),
                      cudaSuccess);
            on_source = heongpu::ExecutionOptions().set_stream(source);
            on_target = heongpu::ExecutionOptions().set_stream(target);

            for (int i = 0; i < slots; i++)
            {
                xs.push_back(0.5 * std::sin(0.01 * i));
                ys.push_back(0.5 * std::cos(0.01 * i));
            }
            x = std::make_unique<Cipher>(
                encryptor->encrypt(encoder->encode(xs)));
            y = std::make_unique<Cipher>(
                encryptor->encrypt(encoder->encode(ys)));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }

        void TearDown() override
        {
            cudaDeviceSynchronize();
            cudaStreamDestroy(source);
            cudaStreamDestroy(target);
        }

        void check(Cipher& result, const std::vector<double>& expected)
        {
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            std::vector<double> values =
                encoder->decode(decryptor->decrypt(result));
            for (int i = 0; i < slots; i++)
            {
                ASSERT_NEAR(values[i], expected[i], 1e-4) << "slot " << i;
            }
        }

        heongpu::HEContext<S> context;
        int slots = 0;
        std::unique_ptr<heongpu::Secretkey<S>> secret_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin_key;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> operators;
        cudaStream_t source = nullptr, target = nullptr;
        heongpu::ExecutionOptions on_source, on_target;
        std::vector<double> xs, ys;
        std::unique_ptr<Cipher> x, y;
    };
} // namespace

TEST_F(CkksStreamOrder, EvaluatePolynomial)
{
    hold(source);
    Cipher doubled(context, on_source);
    operators->add(*x, *x, doubled, on_source);

    Cipher result = operators->evaluate_polynomial(doubled, {0.0, 1.0, 1.0},
                                                   *relin_key, on_target);

    std::vector<double> expected(slots);
    for (int i = 0; i < slots; i++)
    {
        expected[i] = 2.0 * xs[i] + 4.0 * xs[i] * xs[i];
    }
    check(result, expected);
}

TEST_F(CkksStreamOrder, OperandThatNeedsARescaling)
{
    hold(source);
    Cipher square(context, on_source);
    operators->multiply(*x, *x, square, on_source);
    operators->relinearize_inplace(square, *relin_key, on_source);

    Cipher result(context, on_target);
    operators->multiply(square, *y, result, on_target);
    operators->relinearize_inplace(result, *relin_key, on_target);

    std::vector<double> expected(slots);
    for (int i = 0; i < slots; i++)
    {
        expected[i] = xs[i] * xs[i] * ys[i];
    }
    check(result, expected);
}

TEST_F(CkksStreamOrder, OperandBroughtToAnotherLevel)
{
    Cipher square = operators->multiply(*x, *x);
    operators->relinearize_inplace(square, *relin_key);
    operators->rescale_inplace(square);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    hold(source);
    Cipher doubled(context, on_source);
    operators->add(*y, *y, doubled, on_source);

    Cipher result(context, on_target);
    operators->add(square, doubled, result, on_target);

    std::vector<double> expected(slots);
    for (int i = 0; i < slots; i++)
    {
        expected[i] = xs[i] * xs[i] + 2.0 * ys[i];
    }
    check(result, expected);
}
