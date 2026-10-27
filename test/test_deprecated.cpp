// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// The deprecated names still compile and give the results of the names that
// replace them, until 2.0 removes them.

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    template <typename T> std::vector<Data64> data(T& object)
    {
        std::vector<Data64> values;
        object.get_data(values);
        return values;
    }
} // namespace

TEST(Deprecated, CKKS_Rotate_Rows)
{
    heongpu::HEParams<S> params;
    params.multiplicative_depth = 1;
    auto context = params.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    auto galois_key = keygen.galois_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> message(context->get_poly_modulus_degree() / 2);
    for (std::size_t i = 0; i < message.size(); i++)
    {
        message[i] = std::sin(0.01 * double(i));
    }
    auto plain = encoder.encode(message, params.scale());
    auto x = encryptor.encrypt(plain);

    auto expected = operators.rotate(x, galois_key, 3);
    auto old = operators.rotate_rows(x, galois_key, 3);
    EXPECT_EQ(data(old), data(expected));

    heongpu::Ciphertext<S> output(context);
    operators.rotate_rows(x, output, galois_key, 3);
    EXPECT_EQ(data(output), data(expected));

    auto y = x;
    operators.rotate_rows_inplace(y, galois_key, 3);
    EXPECT_EQ(data(y), data(expected));

    auto hoisted = operators.rotate_rows_hoisted(x, galois_key, {3});
    EXPECT_EQ(data(hoisted[0]), data(expected));
}

TEST(Deprecated, Upper_Case_Enumerators)
{
    using heongpu::arithmetic_bootstrapping_type;
    using heongpu::encoding;
    using heongpu::logic_bootstrapping_type;
    using heongpu::storage_type;
    EXPECT_EQ(storage_type::HOST, storage_type::host);
    EXPECT_EQ(storage_type::DEVICE, storage_type::device);
    EXPECT_EQ(encoding::SLOT, encoding::slot);
    EXPECT_EQ(encoding::COEFFICIENT, encoding::coefficient);
    EXPECT_EQ(arithmetic_bootstrapping_type::NONE,
              arithmetic_bootstrapping_type::none);
    EXPECT_EQ(arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING,
              arithmetic_bootstrapping_type::regular);
    EXPECT_EQ(arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING,
              arithmetic_bootstrapping_type::slim);
    EXPECT_EQ(logic_bootstrapping_type::NONE, logic_bootstrapping_type::none);
    EXPECT_EQ(logic_bootstrapping_type::BIT_BOOTSTRAPPING,
              logic_bootstrapping_type::bit);
    EXPECT_EQ(logic_bootstrapping_type::GATE_BOOTSTRAPPING,
              logic_bootstrapping_type::gate);
}
