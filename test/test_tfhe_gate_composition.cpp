// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <functional>
#include <random>
#include <string>
#include <vector>

// The outputs of the TFHE gates are inputs of other gates: every gate gives
// correct bits on two gate outputs, and a ripple carry adder, whose gates
// mostly take gate outputs, adds numbers correctly.

namespace
{
    constexpr auto S = heongpu::Scheme::TFHE;
    using Cipher = heongpu::Ciphertext<S>;
    using Bits = std::vector<bool>;
    constexpr size_t size = 256;

    class TfheGateComposition : public ::testing::Test
    {
      protected:
        static void SetUpTestSuite()
        {
            context = new heongpu::HEContext<S>(heongpu::GenHEContext<S>());
            heongpu::HEKeyGenerator<S> keygen(*context);
            secret_key = new heongpu::Secretkey<S>(*context);
            keygen.generate_secret_key(*secret_key);
            boot_key = new heongpu::Bootstrappingkey<S>(*context);
            keygen.generate_bootstrapping_key(*boot_key, *secret_key);
        }

        static void TearDownTestSuite()
        {
            delete boot_key;
            delete secret_key;
            delete context;
        }

        Bits random_bits()
        {
            std::uniform_int_distribution<int> bit(0, 1);
            Bits bits(size);
            for (size_t i = 0; i < size; i++)
            {
                bits[i] = bit(generator);
            }
            return bits;
        }

        Cipher encrypt(const Bits& bits)
        {
            heongpu::HEEncryptor<S> encryptor(*context, *secret_key);
            Cipher ciphertext(*context);
            encryptor.encrypt(ciphertext, bits);
            return ciphertext;
        }

        Bits decrypt(Cipher& ciphertext)
        {
            heongpu::HEDecryptor<S> decryptor(*context, *secret_key);
            Bits bits;
            decryptor.decrypt(ciphertext, bits);
            return bits;
        }

        static heongpu::HEContext<S>* context;
        static heongpu::Secretkey<S>* secret_key;
        static heongpu::Bootstrappingkey<S>* boot_key;
        std::mt19937 generator{2026};
    };

    heongpu::HEContext<S>* TfheGateComposition::context = nullptr;
    heongpu::Secretkey<S>* TfheGateComposition::secret_key = nullptr;
    heongpu::Bootstrappingkey<S>* TfheGateComposition::boot_key = nullptr;

    using Gate = std::function<void(heongpu::HELogicOperator<S>&, const Cipher&,
                                    const Cipher&, Cipher&,
                                    const heongpu::Bootstrappingkey<S>&)>;

    struct NamedGate
    {
        std::string name;
        Gate gate;
        std::function<bool(bool, bool)> expected;
    };

    std::vector<NamedGate> gates()
    {
        return {
            {"NAND",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.NAND(x, y, r, key); },
             [](bool x, bool y) { return !(x && y); }},
            {"AND",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.AND(x, y, r, key); },
             [](bool x, bool y) { return x && y; }},
            {"NOR",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.NOR(x, y, r, key); },
             [](bool x, bool y) { return !(x || y); }},
            {"OR",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.OR(x, y, r, key); },
             [](bool x, bool y) { return x || y; }},
            {"XNOR",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.XNOR(x, y, r, key); },
             [](bool x, bool y) { return x == y; }},
            {"XOR",
             [](auto& op, auto& x, auto& y, auto& r, auto& key)
             { op.XOR(x, y, r, key); },
             [](bool x, bool y) { return x != y; }},
        };
    }

    int wrong_bits(const Bits& result, const Bits& expected)
    {
        int count = 0;
        for (size_t i = 0; i < expected.size(); i++)
        {
            count += (result[i] != expected[i]);
        }
        return count;
    }
} // namespace

// G(G(x, y), G(z, w)) for every gate G.
TEST_F(TfheGateComposition, GatesOnGateOutputs)
{
    heongpu::HELogicOperator<S> logic(*context);
    Bits x = random_bits(), y = random_bits(), z = random_bits(),
         w = random_bits();
    Cipher cx = encrypt(x), cy = encrypt(y), cz = encrypt(z), cw = encrypt(w);

    for (const auto& g : gates())
    {
        Cipher left(*context), right(*context), output(*context);
        g.gate(logic, cx, cy, left, *boot_key);
        g.gate(logic, cz, cw, right, *boot_key);
        g.gate(logic, left, right, output, *boot_key);

        Bits expected(size);
        for (size_t i = 0; i < size; i++)
        {
            expected[i] =
                g.expected(g.expected(x[i], y[i]), g.expected(z[i], w[i]));
        }
        Bits result = decrypt(output);
        EXPECT_EQ(wrong_bits(result, expected), 0) << g.name;
    }
}

// MUX and NOT with gate outputs as all of their inputs.
TEST_F(TfheGateComposition, MuxAndNotOnGateOutputs)
{
    heongpu::HELogicOperator<S> logic(*context);
    Bits x = random_bits(), y = random_bits(), z = random_bits();
    Cipher cx = encrypt(x), cy = encrypt(y), cz = encrypt(z);

    Cipher a(*context), b(*context), c(*context), output(*context);
    logic.XOR(cx, cy, a, *boot_key);
    logic.AND(cy, cz, b, *boot_key);
    logic.OR(cx, cz, c, *boot_key);
    logic.MUX(a, b, c, output, *boot_key);
    Cipher negated(*context);
    logic.NOT(output, negated);

    Bits expected(size);
    for (size_t i = 0; i < size; i++)
    {
        bool ai = x[i] != y[i], bi = y[i] && z[i], ci = x[i] || z[i];
        expected[i] = !(ci ? ai : bi);
    }
    EXPECT_EQ(wrong_bits(decrypt(negated), expected), 0);
}

// An 8 bit ripple carry adder in every one of the bit positions.
TEST_F(TfheGateComposition, RippleCarryAdder)
{
    constexpr int width = 8;
    heongpu::HELogicOperator<S> logic(*context);

    std::vector<Bits> a(width), b(width);
    std::vector<Cipher> ca, cb;
    for (int j = 0; j < width; j++)
    {
        a[j] = random_bits();
        b[j] = random_bits();
        ca.push_back(encrypt(a[j]));
        cb.push_back(encrypt(b[j]));
    }

    std::vector<Cipher> sum;
    Cipher carry = encrypt(Bits(size, false));
    for (int j = 0; j < width; j++)
    {
        Cipher half(*context), bit(*context), generate(*context),
            propagate(*context), next_carry(*context);
        logic.XOR(ca[j], cb[j], half, *boot_key);
        logic.XOR(half, carry, bit, *boot_key);
        logic.AND(ca[j], cb[j], generate, *boot_key);
        logic.AND(half, carry, propagate, *boot_key);
        logic.OR(generate, propagate, next_carry, *boot_key);
        sum.push_back(std::move(bit));
        carry = std::move(next_carry);
    }

    int wrong = 0;
    std::vector<Bits> result;
    for (int j = 0; j < width; j++)
    {
        result.push_back(decrypt(sum[j]));
    }
    for (size_t i = 0; i < size; i++)
    {
        unsigned x = 0, y = 0, r = 0;
        for (int j = 0; j < width; j++)
        {
            x |= unsigned(a[j][i]) << j;
            y |= unsigned(b[j][i]) << j;
            r |= unsigned(result[j][i]) << j;
        }
        wrong += (r != ((x + y) & ((1u << width) - 1)));
    }
    EXPECT_EQ(wrong, 0);
}
