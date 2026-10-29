// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// Operations leave their inputs unchanged.

namespace
{
    constexpr int degree = 8192;

    template <typename T> std::string bytes(const T& object)
    {
        std::stringstream stream;
        object.save(stream);
        return stream.str();
    }

    struct Snapshot
    {
        std::string bytes;
        bool on_device;
    };

    template <typename T> Snapshot snapshot(const T& object)
    {
        return {bytes(object), object.is_on_device()};
    }

    // Runs operation and checks that the data of inputs and where they are
    // stored did not change.
    template <typename F, typename... T>
    void expect_unchanged(const char* name, F operation, const T&... inputs)
    {
        SCOPED_TRACE(name);
        std::vector<Snapshot> before{snapshot(inputs)...};
        operation();
        std::vector<Snapshot> after{snapshot(inputs)...};
        for (std::size_t i = 0; i < before.size(); i++)
        {
            EXPECT_EQ(after[i].bytes, before[i].bytes) << "input " << i;
            EXPECT_EQ(after[i].on_device, before[i].on_device) << "input " << i;
        }
    }

    // The inputs on the device, in host memory, and in host memory with
    // options that move the objects of an operation to the device.
    enum class placement
    {
        device,
        host,
        host_moving_options
    };

    const placement placements[] = {placement::device, placement::host,
                                    placement::host_moving_options};

    template <typename T> T placed(const T& object, placement where)
    {
        T copy = object;
        if (where != placement::device)
        {
            copy.store_in_host();
            // The copy to host memory is asynchronous.
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        return copy;
    }

    heongpu::ExecutionOptions options_for(placement where)
    {
        heongpu::ExecutionOptions options;
        if (where == placement::host_moving_options)
        {
            options.set_storage_type(heongpu::storage_type::device)
                .set_initial_location(false);
        }
        return options;
    }
} // namespace

TEST(Inputs, BFV_Operations_Leave_Their_Inputs)
{
    constexpr auto S = heongpu::Scheme::BFV;
    using Ciphertext = heongpu::Ciphertext<S>;
    auto context = heongpu::GenHEContext<S>();
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_default_values(1);
    context->set_plain_modulus(1032193);
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    auto relin_key0 = keygen.relin_key(secret_key);
    auto galois_key0 = keygen.galois_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);
    heongpu::HELogicOperator<S> logic(context, encoder);

    std::vector<uint64_t> message(degree);
    for (int i = 0; i < degree; i++)
    {
        message[i] = i % 2;
    }
    auto plain0 = encoder.encode(message);
    auto x0 = encryptor.encrypt(plain0);
    auto y0 = encryptor.encrypt(plain0);

    for (placement where : placements)
    {
        SCOPED_TRACE(static_cast<int>(where));
        const auto options = options_for(where);
        const auto x = placed(x0, where);
        const auto y = placed(y0, where);
        const auto plain = placed(plain0, where);
        const auto relin_key = placed(relin_key0, where);
        const auto galois_key = placed(galois_key0, where);
        Ciphertext out(context);

        expect_unchanged(
            "add", [&] { operators.add(x, y, out, options); }, x, y);
        expect_unchanged(
            "sub", [&] { operators.sub(x, y, out, options); }, x, y);
        expect_unchanged(
            "negate", [&] { operators.negate(x, out, options); }, x);
        expect_unchanged(
            "add_plain", [&] { operators.add_plain(x, plain, out, options); },
            x, plain);
        expect_unchanged(
            "sub_plain", [&] { operators.sub_plain(x, plain, out, options); },
            x, plain);
        expect_unchanged(
            "multiply_plain",
            [&] { operators.multiply_plain(x, plain, out, options); }, x,
            plain);
        expect_unchanged(
            "multiply", [&] { operators.multiply(x, y, out, options); }, x, y);
        expect_unchanged(
            "relinearize_inplace",
            [&]
            {
                operators.multiply(x, y, out, options);
                operators.relinearize_inplace(out, relin_key, options);
            },
            x, y, relin_key);
        expect_unchanged(
            "rotate_rows",
            [&] { operators.rotate_rows(x, out, galois_key, 1, options); }, x,
            galois_key);
        expect_unchanged(
            "rotate_columns",
            [&] { operators.rotate_columns(x, out, galois_key, options); }, x,
            galois_key);
        expect_unchanged(
            "rotate_rows_hoisted",
            [&] {
                operators.rotate_rows_hoisted(x, galois_key, {1, 2}, options);
            },
            x, galois_key);
        expect_unchanged(
            "add_inplace",
            [&]
            {
                Ciphertext z = x;
                operators.add_inplace(z, y, options);
            },
            x, y);
        expect_unchanged(
            "AND", [&] { logic.AND(x, y, out, relin_key, options); }, x, y,
            relin_key);
        expect_unchanged(
            "XOR", [&] { logic.XOR(x, y, out, relin_key, options); }, x, y,
            relin_key);
        expect_unchanged(
            "NOT", [&] { logic.NOT(x, out, options); }, x);
    }
}

TEST(Inputs, BFV_NOT_Leaves_Its_Input)
{
    constexpr auto S = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<S>();
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_default_values(1);
    context->set_plain_modulus(1032193);
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context, encoder);

    std::vector<uint64_t> message(degree);
    for (int i = 0; i < degree; i++)
    {
        message[i] = i % 2;
    }
    auto plain = encoder.encode(message);
    auto input = encryptor.encrypt(plain);
    const std::string before = bytes(input);

    heongpu::Ciphertext<S> output(context);
    logic.NOT(input, output);
    EXPECT_EQ(bytes(input), before);

    auto decrypted = decryptor.decrypt(output);
    std::vector<uint64_t> result;
    encoder.decode(result, decrypted);
    for (int i = 0; i < degree; i++)
    {
        ASSERT_EQ(result[i], 1 - message[i]) << i;
    }
}

TEST(Inputs, CKKS_NOT_Leaves_Its_Input)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    auto context = heongpu::GenHEContext<S>();
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    context->generate();
    const double scale = std::pow(2.0, 30);
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context, encoder, scale);

    std::vector<double> message(degree / 2);
    for (int i = 0; i < degree / 2; i++)
    {
        message[i] = i % 2;
    }
    auto plain = encoder.encode(message, scale);
    auto input = encryptor.encrypt(plain);
    const std::string before = bytes(input);

    heongpu::Ciphertext<S> output(context);
    logic.NOT(input, output);
    EXPECT_EQ(bytes(input), before);

    auto decrypted = decryptor.decrypt(output);
    std::vector<double> result;
    encoder.decode(result, decrypted);
    for (int i = 0; i < degree / 2; i++)
    {
        ASSERT_NEAR(result[i], 1 - message[i], 1e-3) << i;
    }
}

// TFHE MUX takes its control input from host memory too.
TEST(Inputs, TFHE_MUX_With_A_Control_Input_In_Host_Memory)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = heongpu::GenHEContext<S>();
    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Bootstrappingkey<S> boot_key(context);
    keygen.generate_bootstrapping_key(boot_key, secret_key);
    heongpu::HEEncryptor<S> encryptor(context, secret_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context);

    constexpr int size = 32;
    std::vector<bool> a(size), b(size), c(size), expected(size);
    for (int i = 0; i < size; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
        c[i] = (i / 4) % 2;
        expected[i] = c[i] ? a[i] : b[i];
    }
    heongpu::Ciphertext<S> x(context), y(context), control(context);
    encryptor.encrypt(x, a);
    encryptor.encrypt(y, b);
    encryptor.encrypt(control, c);
    control.store_in_host();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    heongpu::Ciphertext<S> output(context);
    logic.MUX(x, y, control, output, boot_key);
    std::vector<bool> result;
    decryptor.decrypt(output, result);
    EXPECT_EQ(result, expected);
}

TEST(Inputs, CKKS_Operations_Leave_Their_Inputs)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Ciphertext = heongpu::Ciphertext<S>;
    using Mode = heongpu::rescale_mode;
    for (Mode mode : {Mode::manual, Mode::automatic})
    {
        SCOPED_TRACE(heongpu::detail::name(mode));
        heongpu::HEParams<S> params;
        params.multiplicative_depth = 3;
        params.rescale_mode = mode;
        auto context = params.make_context();
        context->generate();
        heongpu::HEKeyGenerator<S> keygen(context);
        auto secret_key = keygen.secret_key();
        auto public_key = keygen.public_key(secret_key);
        auto relin_key0 = keygen.relin_key(secret_key);
        auto galois_key0 = keygen.galois_key(secret_key);
        heongpu::HEEncoder<S> encoder(context);
        heongpu::HEEncryptor<S> encryptor(context, public_key);
        heongpu::HEArithmeticOperator<S> operators(context, encoder);

        std::vector<double> message(context->get_poly_modulus_degree() / 2);
        for (std::size_t i = 0; i < message.size(); i++)
        {
            message[i] = std::sin(0.01 * double(i));
        }
        auto plain0 = mode == Mode::automatic
                          ? encoder.encode(message)
                          : encoder.encode(message, params.scale());
        auto x0 = encryptor.encrypt(plain0);
        auto y0 = encryptor.encrypt(plain0);
        // A product that needs a rescaling, for the automatic mode.
        auto z0 = operators.multiply(x0, y0);
        operators.relinearize_inplace(z0, relin_key0);
        auto plain_z0 = mode == Mode::automatic
                            ? encoder.encode(message, z0.depth() + 1)
                            : plain0;

        for (placement where : placements)
        {
            SCOPED_TRACE(static_cast<int>(where));
            const auto options = options_for(where);
            const auto x = placed(x0, where);
            const auto y = placed(y0, where);
            const auto z = placed(z0, where);
            const auto plain = placed(plain0, where);
            const auto plain_z = placed(plain_z0, where);
            const auto relin_key = placed(relin_key0, where);
            const auto galois_key = placed(galois_key0, where);
            Ciphertext out(context);

            expect_unchanged(
                "add", [&] { operators.add(x, y, out, options); }, x, y);
            expect_unchanged(
                "sub", [&] { operators.sub(x, y, out, options); }, x, y);
            expect_unchanged(
                "negate", [&] { operators.negate(x, out, options); }, x);
            expect_unchanged(
                "add_plain",
                [&] { operators.add_plain(x, plain, out, options); }, x, plain);
            expect_unchanged(
                "sub_plain",
                [&] { operators.sub_plain(x, plain, out, options); }, x, plain);
            expect_unchanged(
                "multiply_plain",
                [&] { operators.multiply_plain(x, plain, out, options); }, x,
                plain);
            expect_unchanged(
                "multiply", [&] { operators.multiply(x, y, out, options); }, x,
                y);
            expect_unchanged(
                "mod_drop", [&] { operators.mod_drop(x, out, options); }, x);
            expect_unchanged(
                "rotate",
                [&] { operators.rotate(x, out, galois_key, 1, options); }, x,
                galois_key);
            expect_unchanged(
                "rotate_hoisted",
                [&] {
                    operators.rotate_hoisted(x, galois_key, {1, 2}, options);
                },
                x, galois_key);
            expect_unchanged(
                "evaluate_polynomial",
                [&] {
                    operators.evaluate_polynomial(x, {0.5, 1.0, 0.25},
                                                  relin_key, options);
                },
                x, relin_key);
            expect_unchanged(
                "add_inplace",
                [&]
                {
                    Ciphertext w = x;
                    operators.add_inplace(w, y, options);
                },
                x, y);
            if (mode == Mode::automatic)
            {
                expect_unchanged(
                    "add after rescaling",
                    [&] { operators.add(z, x, out, options); }, z, x);
                expect_unchanged(
                    "multiply after rescaling",
                    [&] { operators.multiply(z, y, out, options); }, z, y);
                expect_unchanged(
                    "add_plain after rescaling",
                    [&] { operators.add_plain(z, plain_z, out, options); }, z,
                    plain_z);
                expect_unchanged(
                    "rotate after rescaling",
                    [&] { operators.rotate(z, out, galois_key, 1, options); },
                    z, galois_key);
            }
        }
    }
}

// In rescale_mode::automatic, add_plain_inplace and sub_plain_inplace of a
// ciphertext that needs a rescaling give the rescaled sum and difference.
TEST(Inputs, CKKS_Automatic_Plain_Inplace_After_Rescaling)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    heongpu::HEParams<S> params;
    params.multiplicative_depth = 3;
    params.rescale_mode = heongpu::rescale_mode::automatic;
    auto context = params.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    auto relin_key = keygen.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    std::vector<double> a(context->get_poly_modulus_degree() / 2), b(a.size());
    for (std::size_t i = 0; i < a.size(); i++)
    {
        a[i] = std::sin(0.01 * double(i));
        b[i] = std::cos(0.02 * double(i));
    }
    auto plain_a = encoder.encode(a);
    auto x = encryptor.encrypt(plain_a);
    auto square = operators.multiply(x, x);
    operators.relinearize_inplace(square, relin_key);
    ASSERT_TRUE(square.rescale_required());
    auto plain_b = encoder.encode(b, square.depth() + 1);

    for (bool add : {true, false})
    {
        auto result = square;
        if (add)
            operators.add_plain_inplace(result, plain_b);
        else
            operators.sub_plain_inplace(result, plain_b);
        EXPECT_FALSE(result.rescale_required());
        auto decrypted = decryptor.decrypt(result);
        std::vector<double> values;
        encoder.decode(values, decrypted);
        for (std::size_t i = 0; i < a.size(); i++)
        {
            double expected = a[i] * a[i] + (add ? b[i] : -b[i]);
            ASSERT_NEAR(values[i], expected, 1e-4) << i;
        }
    }
}

// TFHE gates read inputs in host memory without moving them, also with options
// that move the objects of an operation to the device, and leave their bits.
TEST(Inputs, TFHE_Gates_Leave_Their_Inputs)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    using Ciphertext = heongpu::Ciphertext<S>;
    auto context = heongpu::GenHEContext<S>();
    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Bootstrappingkey<S> boot_key(context);
    keygen.generate_bootstrapping_key(boot_key, secret_key);
    heongpu::HEEncryptor<S> encryptor(context, secret_key);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    heongpu::HELogicOperator<S> logic(context);

    constexpr int size = 32;
    std::vector<bool> a(size), b(size), c(size);
    for (int i = 0; i < size; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
        c[i] = (i / 4) % 2;
    }
    Ciphertext x0(context), y0(context), z0(context);
    encryptor.encrypt(x0, a);
    encryptor.encrypt(y0, b);
    encryptor.encrypt(z0, c);

    auto bits = [&](Ciphertext& ciphertext)
    {
        std::vector<bool> values;
        decryptor.decrypt(ciphertext, values);
        return values;
    };

    for (placement where : placements)
    {
        SCOPED_TRACE(static_cast<int>(where));
        const auto options = options_for(where);
        const Ciphertext x = placed(x0, where);
        const Ciphertext y = placed(y0, where);
        const Ciphertext z = placed(z0, where);
        const bool on_device = where == placement::device;

        std::vector<bool> nand(size), mux(size), negated(size);
        for (int i = 0; i < size; i++)
        {
            nand[i] = !(a[i] && b[i]);
            mux[i] = c[i] ? a[i] : b[i];
            negated[i] = !a[i];
        }
        Ciphertext out(context);
        logic.NAND(x, y, out, boot_key, options);
        EXPECT_EQ(bits(out), nand);
        logic.MUX(x, y, z, out, boot_key, options);
        EXPECT_EQ(bits(out), mux);
        logic.NOT(x, out, options);
        EXPECT_EQ(bits(out), negated);

        EXPECT_EQ(x.is_on_device(), on_device);
        EXPECT_EQ(y.is_on_device(), on_device);
        EXPECT_EQ(z.is_on_device(), on_device);
        Ciphertext x_copy = x, y_copy = y, z_copy = z;
        EXPECT_EQ(bits(x_copy), a);
        EXPECT_EQ(bits(y_copy), b);
        EXPECT_EQ(bits(z_copy), c);
    }
}

// Encoding, encryption, decryption and key generation leave their inputs,
// also keys in host memory given to an encryptor or decryptor.
template <typename T> class InputsOfSchemes : public ::testing::Test
{
};
using SchemeTypes = ::testing::Types<
    std::integral_constant<heongpu::Scheme, heongpu::Scheme::BFV>,
    std::integral_constant<heongpu::Scheme, heongpu::Scheme::CKKS>>;
TYPED_TEST_SUITE(InputsOfSchemes, SchemeTypes);

TYPED_TEST(InputsOfSchemes, Encryption_Decryption_And_Keys_Leave_Their_Inputs)
{
    constexpr auto S = TypeParam::value;
    heongpu::HEParams<S> params;
    params.multiplicative_depth = 2;
    if constexpr (S == heongpu::Scheme::BFV)
    {
        params.plain_modulus_bit_size = 20;
    }
    auto context = params.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key0 = keygen.secret_key();
    auto public_key0 = keygen.public_key(secret_key0);
    heongpu::HEEncoder<S> encoder(context);
    const std::size_t n = context->get_poly_modulus_degree();
    auto plain0 = [&]
    {
        if constexpr (S == heongpu::Scheme::BFV)
            return encoder.encode(std::vector<uint64_t>(n, 3));
        else
            return encoder.encode(std::vector<double>(n / 2, 0.5),
                                  params.scale());
    }();

    for (placement where : placements)
    {
        SCOPED_TRACE(static_cast<int>(where));
        const auto options = options_for(where);
        const auto secret_key = placed(secret_key0, where);
        const auto public_key = placed(public_key0, where);
        const auto plain = placed(plain0, where);

        expect_unchanged(
            "keys",
            [&]
            {
                heongpu::Publickey<S> pk(context);
                keygen.generate_public_key(pk, secret_key, options);
                heongpu::Relinkey<S> rk(context);
                keygen.generate_relin_key(rk, secret_key, options);
            },
            secret_key);
        heongpu::Ciphertext<S> ciphertext0(context);
        expect_unchanged(
            "encryption",
            [&]
            {
                heongpu::HEEncryptor<S> encryptor(context, public_key);
                encryptor.encrypt(ciphertext0, plain, options);
            },
            public_key, plain);
        const auto ciphertext = placed(ciphertext0, where);
        heongpu::Plaintext<S> decrypted(context);
        expect_unchanged(
            "decryption",
            [&]
            {
                heongpu::HEDecryptor<S> decryptor(context, secret_key);
                decryptor.decrypt(decrypted, ciphertext, options);
            },
            secret_key, ciphertext);
        const auto decrypted_placed = placed(decrypted, where);
        expect_unchanged(
            "decoding",
            [&]
            {
                if constexpr (S == heongpu::Scheme::BFV)
                {
                    std::vector<uint64_t> values;
                    encoder.decode(values, decrypted_placed, options);
                    EXPECT_EQ(values[0], 3u);
                }
                else
                {
                    std::vector<double> values;
                    encoder.decode(values, decrypted_placed, options);
                    EXPECT_NEAR(values[0], 0.5, 1e-3);
                }
            },
            decrypted_placed);
    }
}
