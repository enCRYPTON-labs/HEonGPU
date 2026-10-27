// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// rotate_hoisted (rotate_rows_hoisted for BFV) gives the results of rotate
// (rotate_rows) for every shift, with shifts that have their own key, shifts
// composed from the power of two keys and shifts that do not move the slots.

namespace
{
    constexpr int degree = 8192;

    template <typename T> std::vector<Data64> data(T& object)
    {
        std::vector<Data64> values;
        object.get_data(values);
        return values;
    }

    // The rotations are rotate for CKKS and rotate_rows for BFV.
    template <heongpu::Scheme S>
    std::vector<heongpu::Ciphertext<S>>
    hoisted(heongpu::HEArithmeticOperator<S>& operators,
            heongpu::Ciphertext<S>& input, heongpu::Galoiskey<S>& key,
            const std::vector<int>& shifts)
    {
        if constexpr (S == heongpu::Scheme::CKKS)
        {
            return operators.rotate_hoisted(input, key, shifts);
        }
        else
        {
            return operators.rotate_rows_hoisted(input, key, shifts);
        }
    }

    template <heongpu::Scheme S>
    void rotate(heongpu::HEArithmeticOperator<S>& operators,
                heongpu::Ciphertext<S>& input, heongpu::Ciphertext<S>& output,
                heongpu::Galoiskey<S>& key, int shift)
    {
        if constexpr (S == heongpu::Scheme::CKKS)
        {
            operators.rotate(input, output, key, shift);
        }
        else
        {
            operators.rotate_rows(input, output, key, shift);
        }
    }

    template <heongpu::Scheme S> heongpu::HEContext<S> make_context(int p_count)
    {
        auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(degree);
        if constexpr (S == heongpu::Scheme::CKKS)
        {
            context->set_coeff_modulus_bit_sizes({50, 40, 40, 40},
                                                 std::vector<int>(p_count, 60));
        }
        else
        {
            context->set_coeff_modulus_bit_sizes({54, 54, 54},
                                                 std::vector<int>(p_count, 55));
            context->set_plain_modulus(1032193);
        }
        context->generate();
        return context;
    }

    template <heongpu::Scheme S> void check(int p_count, bool host_key)
    {
        auto context = make_context<S>(p_count);
        heongpu::HEKeyGenerator<S> generator(context);
        auto secret_key = generator.secret_key();
        auto public_key = generator.public_key(secret_key);
        // Keys for these shifts and for none of 2 and 100.
        std::vector<int> key_shifts = {1, 3, 5, -1, -7, 64};
        auto galois_key = generator.galois_key(secret_key, key_shifts);
        // The power of two keys, from which the rotations compose shifts.
        auto power_key = generator.galois_key(secret_key);
        if (host_key)
        {
            galois_key.store_in_host();
        }

        heongpu::HEEncoder<S> encoder(context);
        heongpu::HEEncryptor<S> encryptor(context, public_key);
        heongpu::HEArithmeticOperator<S> operators(context, encoder);
        heongpu::Plaintext<S> plaintext(context);
        if constexpr (S == heongpu::Scheme::CKKS)
        {
            std::vector<double> message(degree / 2);
            for (int i = 0; i < degree / 2; i++)
            {
                message[i] = std::sin(0.01 * i);
            }
            encoder.encode(plaintext, message, std::pow(2.0, 40));
        }
        else
        {
            std::vector<uint64_t> message(degree);
            for (int i = 0; i < degree; i++)
            {
                message[i] = i % 1000;
            }
            encoder.encode(plaintext, message);
        }
        auto ciphertext = encryptor.encrypt(plaintext);

        std::vector<int> shifts = {1, 3, 5, -1, -7, 64, 0, degree / 2};
        auto results = hoisted(operators, ciphertext, galois_key, shifts);
        ASSERT_EQ(results.size(), shifts.size());
        for (size_t i = 0; i < shifts.size(); i++)
        {
            heongpu::Ciphertext<S> expected(context);
            rotate(operators, ciphertext, expected, galois_key, shifts[i]);
            EXPECT_EQ(data(results[i]), data(expected))
                << "shift " << shifts[i];
        }

        // Shifts without their own key are composed from the power of two
        // keys, as rotate does.
        std::vector<int> composed = {2, 3, 100, -5};
        auto composed_results =
            hoisted(operators, ciphertext, power_key, composed);
        for (size_t i = 0; i < composed.size(); i++)
        {
            heongpu::Ciphertext<S> expected(context);
            rotate(operators, ciphertext, expected, power_key, composed[i]);
            EXPECT_EQ(data(composed_results[i]), data(expected))
                << "shift " << composed[i];
        }

        if constexpr (S == heongpu::Scheme::CKKS)
        {
            // Below the top level.
            operators.mod_drop_inplace(ciphertext);
            auto dropped =
                operators.rotate_hoisted(ciphertext, galois_key, {1, -7});
            heongpu::Ciphertext<S> expected(context);
            operators.rotate(ciphertext, expected, galois_key, -7);
            EXPECT_EQ(data(dropped[1]), data(expected));
            EXPECT_EQ(dropped[1].depth(), 1);
        }
    }
} // namespace

TEST(HEonGPU, CKKS_Rotate_Hoisted)
{
    check<heongpu::Scheme::CKKS>(1, false);
    check<heongpu::Scheme::CKKS>(2, false);
    check<heongpu::Scheme::CKKS>(2, true);
}

TEST(HEonGPU, BFV_Rotate_Rows_Hoisted)
{
    check<heongpu::Scheme::BFV>(1, false);
    check<heongpu::Scheme::BFV>(2, false);
    check<heongpu::Scheme::BFV>(2, true);
}
