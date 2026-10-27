// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>

// The format v1, deprecated; its writing is tested until 2.0 removes it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static constexpr auto format_v1 = heongpu::serialization_format::v1;
#pragma GCC diagnostic pop

namespace
{
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;

    heongpu::HEContext<scheme> make_context(int n = degree)
    {
        auto context =
            heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(n);
        context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
        context->generate();
        return context;
    }

    // Runs function, expects it to throw E and returns the message.
    template <typename E, typename F> std::string message_of(F function)
    {
        try
        {
            function();
        }
        catch (const E& error)
        {
            return error.what();
        }
        catch (const std::exception& error)
        {
            ADD_FAILURE() << "Unexpected exception type: " << error.what();
            return "";
        }
        ADD_FAILURE() << "No exception was thrown";
        return "";
    }

    bool contains(const std::string& text, const std::string& part)
    {
        return text.find(part) != std::string::npos;
    }

    class CKKSErrors : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            context = make_context();
            heongpu::HEKeyGenerator<scheme> generator(context);
            secret_key = heongpu::Secretkey<scheme>(context);
            generator.generate_secret_key(secret_key);
            public_key = heongpu::Publickey<scheme>(context);
            generator.generate_public_key(public_key, secret_key);
        }

        heongpu::Ciphertext<scheme> encrypt(double value)
        {
            heongpu::HEEncoder<scheme> encoder(context);
            heongpu::HEEncryptor<scheme> encryptor(context, public_key);
            heongpu::Plaintext<scheme> plaintext(context);
            encoder.encode(plaintext, std::vector<double>(degree / 2, value),
                           scale);
            heongpu::Ciphertext<scheme> ciphertext(context);
            encryptor.encrypt(ciphertext, plaintext);
            return ciphertext;
        }

        const double scale = pow(2.0, 30);
        heongpu::HEContext<scheme> context;
        heongpu::Secretkey<scheme> secret_key;
        heongpu::Publickey<scheme> public_key;
    };
} // namespace

TEST(HEonGPU, Error_Types_Keep_Standard_Base_Classes)
{
    static_assert(
        std::is_base_of_v<std::invalid_argument, heongpu::ParameterError>);
    static_assert(
        std::is_base_of_v<std::invalid_argument, heongpu::StateError>);
    static_assert(std::is_base_of_v<std::invalid_argument,
                                    heongpu::ContextMismatchError>);
    static_assert(
        std::is_base_of_v<std::invalid_argument, heongpu::LevelMismatchError>);
    static_assert(
        std::is_base_of_v<std::invalid_argument, heongpu::KeyNotFoundError>);
    static_assert(
        std::is_base_of_v<std::runtime_error, heongpu::SerializationError>);
    static_assert(std::is_base_of_v<std::runtime_error, heongpu::SystemError>);

    static_assert(std::is_base_of_v<heongpu::Error, heongpu::ParameterError>);
    static_assert(std::is_base_of_v<heongpu::Error, heongpu::StateError>);
    static_assert(
        std::is_base_of_v<heongpu::Error, heongpu::ContextMismatchError>);
    static_assert(
        std::is_base_of_v<heongpu::Error, heongpu::LevelMismatchError>);
    static_assert(std::is_base_of_v<heongpu::Error, heongpu::KeyNotFoundError>);
    static_assert(
        std::is_base_of_v<heongpu::Error, heongpu::SerializationError>);
    static_assert(std::is_base_of_v<heongpu::Error, heongpu::SystemError>);

    // what() is the same through both bases.
    const heongpu::SystemError error("system failure");
    const heongpu::Error& base = error;
    const std::runtime_error& standard = error;
    EXPECT_STREQ(base.what(), "system failure");
    EXPECT_STREQ(standard.what(), "system failure");
}

TEST(HEonGPU, Error_Message_Builder)
{
    EXPECT_EQ(heongpu::error_message("depth ", 1, " and ", 2, "."),
              "depth 1 and 2.");
}

TEST_F(CKKSErrors, ParameterError_Reports_The_Scale)
{
    heongpu::HEEncoder<scheme> encoder(context);
    heongpu::Plaintext<scheme> plaintext(context);
    std::vector<double> message(degree / 2, 0.5);
    const std::string message_text = message_of<heongpu::ParameterError>(
        [&] { encoder.encode(plaintext, message, pow(2.0, 200)); });
    EXPECT_TRUE(contains(message_text, "Scale out of bounds!"));
    EXPECT_TRUE(contains(message_text, "2^140")) << message_text;

    // Existing handlers for the standard type still catch it.
    EXPECT_THROW(encoder.encode(plaintext, message, -1.0),
                 std::invalid_argument);
}

TEST_F(CKKSErrors, StateError_For_A_Secretkey_That_Is_Not_Generated)
{
    heongpu::Secretkey<scheme> empty_key(context);
    const std::string message_text = message_of<heongpu::StateError>(
        [&] { heongpu::HEDecryptor<scheme> decryptor(context, empty_key); });
    EXPECT_TRUE(contains(message_text, "Secretkey is not generated!"));
    EXPECT_TRUE(contains(message_text, "generate_secret_key"));

    EXPECT_THROW(heongpu::HEDecryptor<scheme>(context, empty_key),
                 std::logic_error);
}

TEST_F(CKKSErrors, ContextMismatchError_Reports_Both_Ring_Sizes)
{
    auto other = make_context(2 * degree);
    const std::string message_text = message_of<heongpu::ContextMismatchError>(
        [&] { heongpu::HEDecryptor<scheme> decryptor(other, secret_key); });
    EXPECT_TRUE(contains(message_text, "Secretkey context mismatch!"));
    EXPECT_TRUE(contains(message_text, "ring size 4096")) << message_text;
    EXPECT_TRUE(contains(message_text, "ring size 8192")) << message_text;
}

TEST_F(CKKSErrors, LevelMismatchError_Reports_Both_Depths)
{
    heongpu::HEEncoder<scheme> encoder(context);
    heongpu::HEArithmeticOperator<scheme> operators(context, encoder);
    auto input1 = encrypt(0.25);
    auto input2 = encrypt(0.5);
    operators.mod_drop_inplace(input2);

    heongpu::Ciphertext<scheme> output(context);
    const std::string message_text = message_of<heongpu::LevelMismatchError>(
        [&] { operators.add(input1, input2, output); });
    EXPECT_TRUE(contains(message_text, "Ciphertexts leveled are not equal"));
    EXPECT_TRUE(contains(message_text, "depth 0 and 1")) << message_text;
    EXPECT_TRUE(contains(message_text, "mod_drop"));

    EXPECT_THROW(operators.add(input1, input2, output), std::logic_error);
}

TEST(HEonGPU, ParameterError_Reports_The_Key_Switching_Digit)
{
    // P has 50 bits, the digits of one Q prime 40 bits: valid.
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
        heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    EXPECT_NO_THROW(context->set_coeff_modulus_bit_sizes({40, 40, 40}, {50}));

    // Digits of two Q primes have 70 and 40 bits, P 60.
    auto bfv = heongpu::GenHEContext<heongpu::Scheme::BFV>(
        heongpu::sec_level_type::none);
    bfv->set_poly_modulus_degree(degree);
    std::string message = message_of<heongpu::ParameterError>(
        [&] {
            bfv->set_coeff_modulus_bit_sizes({35, 35, 40}, {30, 30});
        });
    EXPECT_TRUE(contains(message, "Q primes 0 to 1 has 70 bits")) << message;
    EXPECT_TRUE(contains(message, "60 bits of P")) << message;
}

TEST(HEonGPU, ContextMismatchError_For_A_Key_Of_Other_Digits)
{
    // A key whose key switching digits differ from those of the context, as
    // BFV keys of more than two P primes saved by HEonGPU 1.7.0 or earlier.
    constexpr auto bfv = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<bfv>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36, 36, 36}, {37, 37, 37});
    context->set_plain_modulus(1032193);
    context->generate();
    heongpu::HEKeyGenerator<bfv> generator(context);
    auto secret_key = generator.secret_key();
    auto relin_key = generator.relin_key(secret_key);

    // The format v1 has the fields only; the digit count follows the scheme,
    // the key switching type and three ints.
    std::stringstream stream;
    relin_key.save(stream, format_v1);
    std::string binary = stream.str();
    int digits;
    std::memcpy(&digits, binary.data() + 14, sizeof(int));
    EXPECT_EQ(digits, 2);
    digits = 3;
    std::memcpy(binary.data() + 14, &digits, sizeof(int));
    std::stringstream changed(binary);
    heongpu::Relinkey<bfv> loaded;
    loaded.load(changed);

    heongpu::HEEncoder<bfv> encoder(context);
    heongpu::HEArithmeticOperator<bfv> operators(context, encoder);
    auto public_key = generator.public_key(secret_key);
    heongpu::HEEncryptor<bfv> encryptor(context, public_key);
    auto plain = encoder.encode(std::vector<uint64_t>(degree, 3));
    auto x = encryptor.encrypt(plain);
    auto product = operators.multiply(x, x);
    std::string message = message_of<heongpu::ContextMismatchError>(
        [&] { operators.relinearize_inplace(product, loaded); });
    EXPECT_TRUE(contains(message, "has 3 key switching digits")) << message;
    EXPECT_TRUE(contains(message, "splits Q into 2")) << message;
}

TEST(HEonGPU, KeyNotFoundError_Reports_The_Missing_Step)
{
    constexpr auto bfv = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<bfv>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36}, {37});
    context->set_plain_modulus(1032193);
    context->generate();

    heongpu::HEKeyGenerator<bfv> generator(context);
    heongpu::Secretkey<bfv> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<bfv> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    // Shift 3 needs the keys for the steps 1 and 2.
    std::vector<int> shifts = {1};
    heongpu::Galoiskey<bfv> galois_key(context, shifts);
    generator.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<bfv> encoder(context);
    heongpu::HEEncryptor<bfv> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<bfv> operators(context, encoder);
    heongpu::Plaintext<bfv> plaintext(context);
    encoder.encode(plaintext, std::vector<uint64_t>(degree, 1));
    heongpu::Ciphertext<bfv> input(context);
    encryptor.encrypt(input, plaintext);

    heongpu::Ciphertext<bfv> output(context);
    const std::string message_text = message_of<heongpu::KeyNotFoundError>(
        [&] { operators.rotate_rows(input, output, galois_key, 3); });
    EXPECT_TRUE(contains(message_text, "Galois key not present"));
    EXPECT_TRUE(contains(message_text, "rotation step 2")) << message_text;
    EXPECT_TRUE(contains(message_text, "shift 3")) << message_text;

    EXPECT_THROW(operators.rotate_rows(input, output, galois_key, 3),
                 std::logic_error);
}

TEST_F(CKKSErrors, SerializationError_Reports_The_Scheme)
{
    heongpu::HEEncoder<scheme> encoder(context);
    heongpu::Plaintext<scheme> plaintext(context);
    encoder.encode(plaintext, std::vector<double>(degree / 2, 0.5), scale);
    std::stringstream stream;
    plaintext.save(stream, format_v1);
    std::string binary = stream.str();
    binary[0] = static_cast<char>(heongpu::scheme_type::bfv);

    std::stringstream corrupted(binary);
    heongpu::Plaintext<scheme> loaded;
    const std::string message_text = message_of<heongpu::SerializationError>(
        [&] { loaded.load(corrupted); });
    EXPECT_TRUE(contains(message_text, "Invalid scheme binary!"));
    EXPECT_TRUE(contains(message_text, "Expected CKKS")) << message_text;
    EXPECT_TRUE(contains(message_text, "scheme id 1")) << message_text;
}

TEST_F(CKKSErrors, All_HEonGPU_Errors_Share_One_Base)
{
    heongpu::HEEncoder<scheme> encoder(context);
    heongpu::Plaintext<scheme> plaintext(context);
    std::vector<double> message(degree / 2, 0.5);
    bool caught = false;
    try
    {
        encoder.encode(plaintext, message, pow(2.0, 200));
    }
    catch (const heongpu::Error& error)
    {
        caught = contains(error.what(), "Scale out of bounds!");
    }
    EXPECT_TRUE(caught);
}
