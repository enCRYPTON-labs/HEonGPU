// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// The format v1, deprecated; its writing is tested until 2.0 removes it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static constexpr auto format_v1 = heongpu::serialization_format::v1;
#pragma GCC diagnostic pop

// The rotation is rotate for CKKS and rotate_rows for BFV.
template <heongpu::Scheme S>
static void rotate(heongpu::HEArithmeticOperator<S>& operators,
                   heongpu::Ciphertext<S>& input,
                   heongpu::Ciphertext<S>& output, heongpu::Galoiskey<S>& key,
                   int shift)
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

// The serialization_format::v2 header and the compatibility with the format
// v1.

namespace
{
    constexpr int degree = 4096;
    constexpr size_t header_size = 16;

    template <typename T>
    std::string save(const T& object, heongpu::serialization_format format =
                                          heongpu::serialization_format::v2)
    {
        std::stringstream stream;
        object.save(stream, format);
        return stream.str();
    }

    template <typename T> void load(T& object, const std::string& binary)
    {
        std::stringstream stream(binary);
        object.load(stream);
    }

    uint64_t read_id(const std::string& binary)
    {
        uint64_t id = 0;
        for (int i = 0; i < 8; i++)
        {
            id |= uint64_t(static_cast<unsigned char>(binary[8 + i]))
                  << (8 * i);
        }
        return id;
    }

    template <heongpu::Scheme S> heongpu::HEContext<S> make_context(int set)
    {
        auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(degree);
        if constexpr (S == heongpu::Scheme::CKKS)
        {
            context->set_coeff_modulus_bit_sizes(
                set == 0 ? std::vector<int>{40, 30, 30}
                         : std::vector<int>{40, 31, 31},
                {40});
        }
        else
        {
            context->set_coeff_modulus_bit_sizes(
                set == 0 ? std::vector<int>{36, 36} : std::vector<int>{35, 35},
                {37});
            context->set_plain_modulus(1032193);
        }
        context->generate();
        return context;
    }

    template <heongpu::Scheme S> struct Party
    {
        explicit Party(heongpu::HEContext<S> context_)
            : context(context_), encoder(context), operators(context, encoder)
        {
            heongpu::HEKeyGenerator<S> generator(context);
            secret_key = heongpu::Secretkey<S>(context);
            generator.generate_secret_key(secret_key);
            public_key = heongpu::Publickey<S>(context);
            generator.generate_public_key(public_key, secret_key);
            relin_key = heongpu::Relinkey<S>(context);
            generator.generate_relin_key(relin_key, secret_key);
            galois_key = heongpu::Galoiskey<S>(context);
            generator.generate_galois_key(galois_key, secret_key);
            switch_key = heongpu::Switchkey<S>(context);
            heongpu::Secretkey<S> other_key(context);
            generator.generate_secret_key(other_key);
            generator.generate_switch_key(switch_key, other_key, secret_key);
        }

        std::vector<uint64_t> values() const
        {
            return std::vector<uint64_t>(degree, 7);
        }

        heongpu::Plaintext<S> encode()
        {
            heongpu::Plaintext<S> plaintext(context);
            if constexpr (S == heongpu::Scheme::CKKS)
            {
                encoder.encode(plaintext, std::vector<double>(degree / 2, 0.25),
                               pow(2.0, 30));
            }
            else
            {
                encoder.encode(plaintext, values());
            }
            return plaintext;
        }

        heongpu::Ciphertext<S> encrypt()
        {
            auto plaintext = encode();
            heongpu::HEEncryptor<S> encryptor(context, public_key);
            heongpu::Ciphertext<S> ciphertext(context);
            encryptor.encrypt(ciphertext, plaintext);
            return ciphertext;
        }

        // Decrypts and checks the value written by encrypt().
        bool decrypts_correctly(heongpu::Ciphertext<S>& ciphertext)
        {
            heongpu::HEDecryptor<S> decryptor(context, secret_key);
            heongpu::Plaintext<S> plaintext(context);
            decryptor.decrypt(plaintext, ciphertext);
            if constexpr (S == heongpu::Scheme::CKKS)
            {
                std::vector<double> result;
                encoder.decode(result, plaintext);
                for (double value : result)
                {
                    if (std::fabs(value - 0.25) > 1e-3)
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                std::vector<uint64_t> result;
                encoder.decode(result, plaintext);
                return result == values();
            }
        }

        heongpu::HEContext<S> context;
        heongpu::HEEncoder<S> encoder;
        heongpu::HEArithmeticOperator<S> operators;
        heongpu::Secretkey<S> secret_key;
        heongpu::Publickey<S> public_key;
        heongpu::Relinkey<S> relin_key;
        heongpu::Galoiskey<S> galois_key;
        heongpu::Switchkey<S> switch_key;
    };

    template <typename T> class SerializationFormat : public ::testing::Test
    {
      protected:
        static constexpr heongpu::Scheme S = T::value;
        static constexpr heongpu::scheme_type scheme_id =
            S == heongpu::Scheme::CKKS ? heongpu::scheme_type::ckks
                                       : heongpu::scheme_type::bfv;
        Party<S> a{make_context<S>(0)};

        // Checks the v2 header and that the v1 data is the v2 data without
        // its header.
        template <typename O>
        void expect_formats(const O& object, uint8_t object_type,
                            uint64_t context_id)
        {
            const std::string v2 = save(object);
            const std::string v1 = save(object, format_v1);
            ASSERT_GT(v2.size(), header_size);
            EXPECT_EQ(v2.substr(0, 4), "HEGP");
            EXPECT_EQ(static_cast<unsigned char>(v2[4]), 2);
            EXPECT_EQ(static_cast<unsigned char>(v2[5]), 0);
            EXPECT_EQ(static_cast<unsigned char>(v2[6]), object_type);
            EXPECT_EQ(static_cast<unsigned char>(v2[7]),
                      static_cast<unsigned char>(scheme_id));
            EXPECT_EQ(read_id(v2), context_id);
            EXPECT_EQ(v2.substr(header_size), v1);
        }
    };

    template <heongpu::Scheme V>
    using SchemeValue = std::integral_constant<heongpu::Scheme, V>;
    using Schemes = ::testing::Types<SchemeValue<heongpu::Scheme::CKKS>,
                                     SchemeValue<heongpu::Scheme::BFV>>;
} // namespace

TYPED_TEST_SUITE(SerializationFormat, Schemes);

TYPED_TEST(SerializationFormat, Every_Object_Has_A_Header_Before_The_V1_Data)
{
    auto& a = this->a;
    const uint64_t id = a.context->context_id();
    this->expect_formats(*a.context, 1, id);
    this->expect_formats(a.secret_key, 2, id);
    this->expect_formats(a.public_key, 3, id);
    this->expect_formats(a.relin_key, 4, id);
    this->expect_formats(a.galois_key, 5, id);
    this->expect_formats(a.switch_key, 6, id);
    this->expect_formats(a.encode(), 7, id);
    this->expect_formats(a.encrypt(), 8, id);
}

TYPED_TEST(SerializationFormat, V1_And_V2_Data_Load)
{
    constexpr auto S = TestFixture::S;
    auto& a = this->a;
    auto ciphertext = a.encrypt();
    for (auto format : {format_v1, heongpu::serialization_format::v2})
    {
        heongpu::Ciphertext<S> loaded;
        load(loaded, save(ciphertext, format));
        EXPECT_TRUE(a.decrypts_correctly(loaded));

        heongpu::Ciphertext<S> bound(a.context);
        load(bound, save(ciphertext, format));
        EXPECT_TRUE(a.decrypts_correctly(bound));

        auto context = heongpu::GenHEContext<S>();
        load(*context, save(*a.context, format));
        EXPECT_EQ(context->context_id(), a.context->context_id());
    }
}

TYPED_TEST(SerializationFormat, Serializer_Keeps_The_Context)
{
    constexpr auto S = TestFixture::S;
    Party<S> b{make_context<S>(1)};
    auto buffer = heongpu::serializer::serialize(this->a.encrypt());
    auto loaded =
        heongpu::serializer::deserialize<heongpu::Ciphertext<S>>(buffer);
    EXPECT_TRUE(this->a.decrypts_correctly(loaded));

    heongpu::HEDecryptor<S> decryptor(b.context, b.secret_key);
    heongpu::Plaintext<S> plaintext(b.context);
    EXPECT_THROW(decryptor.decrypt(plaintext, loaded),
                 heongpu::ContextMismatchError);
}

TYPED_TEST(SerializationFormat, Invalid_Headers_Are_Rejected)
{
    constexpr auto S = TestFixture::S;
    const std::string binary = save(this->a.encrypt());
    auto expect_rejected = [&](std::string data)
    {
        heongpu::Ciphertext<S> loaded;
        EXPECT_THROW(load(loaded, data), heongpu::SerializationError);
    };

    std::string magic = binary;
    magic[3] = 'X';
    expect_rejected(magic);

    std::string version = binary;
    version[4] = 3;
    expect_rejected(version);

    expect_rejected(binary.substr(0, 10));

    // A plaintext is not a ciphertext.
    expect_rejected(save(this->a.encode()));

    // The scheme of the header must match.
    std::string other_scheme = binary;
    other_scheme[7] = static_cast<char>(S == heongpu::Scheme::CKKS
                                            ? heongpu::scheme_type::bfv
                                            : heongpu::scheme_type::ckks);
    expect_rejected(other_scheme);
}

TYPED_TEST(SerializationFormat, Corrupted_Context_Is_Rejected)
{
    constexpr auto S = TestFixture::S;
    std::string binary = save(*this->a.context);
    binary[8] ^= 1;
    auto context = heongpu::GenHEContext<S>();
    EXPECT_THROW(load(*context, binary), heongpu::SerializationError);
}

TYPED_TEST(SerializationFormat, Unknown_Format_Is_Rejected)
{
    std::stringstream stream;
    EXPECT_THROW(this->a.encrypt().save(
                     stream, static_cast<heongpu::serialization_format>(3)),
                 heongpu::ParameterError);
}

TYPED_TEST(SerializationFormat, Loaded_Keys_Give_The_Same_Results)
{
    constexpr auto S = TestFixture::S;
    using Ciphertext = heongpu::Ciphertext<S>;
    auto& a = this->a;
    auto ciphertext = a.encrypt();

    auto data = [](Ciphertext& c)
    {
        std::vector<Data64> values;
        c.get_data(values);
        return values;
    };
    auto expect_same = [&](auto operation, auto& key)
    {
        std::decay_t<decltype(key)> loaded;
        load(loaded, save(key));
        Ciphertext expected(a.context), result(a.context);
        operation(key, expected);
        operation(loaded, result);
        EXPECT_EQ(data(result), data(expected));
    };

    expect_same(
        [&](heongpu::Relinkey<S>& key, Ciphertext& output)
        {
            a.operators.multiply(ciphertext, ciphertext, output);
            a.operators.relinearize_inplace(output, key);
        },
        a.relin_key);
    expect_same([&](heongpu::Galoiskey<S>& key, Ciphertext& output)
                { rotate(a.operators, ciphertext, output, key, 1); },
                a.galois_key);
    expect_same([&](heongpu::Switchkey<S>& key, Ciphertext& output)
                { a.operators.keyswitch(ciphertext, output, key); },
                a.switch_key);
}

// Earlier versions wrote KEYSWITCHING_METHOD_I for contexts with one P prime
// and for their keys; such data loads and gives the same results.
TYPED_TEST(SerializationFormat, Method_I_Data_Of_Earlier_Versions_Loads)
{
    constexpr auto S = TestFixture::S;
    using Ciphertext = heongpu::Ciphertext<S>;
    auto& a = this->a;
    auto ciphertext = a.encrypt();
    constexpr char method_I = 1, method_II = 2;

    std::string context_binary = save(*a.context);
    ASSERT_EQ(context_binary[header_size + 2], method_II);
    context_binary[header_size + 2] = method_I;
    auto context = heongpu::GenHEContext<S>();
    load(*context, context_binary);
    EXPECT_EQ(context->to_string(), a.context->to_string());

    auto data = [](Ciphertext& c)
    {
        std::vector<Data64> values;
        c.get_data(values);
        return values;
    };
    auto expect_same = [&](auto operation, auto& key)
    {
        std::string binary = save(key);
        ASSERT_EQ(binary[header_size + 1], method_II);
        binary[header_size + 1] = method_I;
        std::decay_t<decltype(key)> loaded;
        load(loaded, binary);
        Ciphertext expected(a.context), result(a.context);
        operation(key, expected);
        operation(loaded, result);
        EXPECT_EQ(data(result), data(expected));
    };

    expect_same(
        [&](heongpu::Relinkey<S>& key, Ciphertext& output)
        {
            a.operators.multiply(ciphertext, ciphertext, output);
            a.operators.relinearize_inplace(output, key);
        },
        a.relin_key);
    expect_same([&](heongpu::Galoiskey<S>& key, Ciphertext& output)
                { rotate(a.operators, ciphertext, output, key, 1); },
                a.galois_key);
    expect_same([&](heongpu::Switchkey<S>& key, Ciphertext& output)
                { a.operators.keyswitch(ciphertext, output, key); },
                a.switch_key);
}
