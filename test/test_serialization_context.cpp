// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

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

// Serialized objects remember the context they were created with: loading
// them for, or using them with, a context with the same ring size and number
// of moduli but different primes must fail.

namespace
{
    constexpr int degree = 4096;

    template <typename T> std::string save(const T& object)
    {
        std::stringstream stream;
        object.save(stream);
        return stream.str();
    }

    template <typename T> void load(T& object, const std::string& binary)
    {
        std::stringstream stream(binary);
        object.load(stream);
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
        }

        heongpu::Plaintext<S> encode(double value)
        {
            heongpu::Plaintext<S> plaintext(context);
            if constexpr (S == heongpu::Scheme::CKKS)
            {
                encoder.encode(plaintext,
                               std::vector<double>(degree / 2, value),
                               pow(2.0, 30));
            }
            else
            {
                encoder.encode(plaintext,
                               std::vector<uint64_t>(
                                   degree, static_cast<uint64_t>(value)));
            }
            return plaintext;
        }

        heongpu::Ciphertext<S> encrypt(double value)
        {
            auto plaintext = encode(value);
            heongpu::HEEncryptor<S> encryptor(context, public_key);
            heongpu::Ciphertext<S> ciphertext(context);
            encryptor.encrypt(ciphertext, plaintext);
            return ciphertext;
        }

        heongpu::HEContext<S> context;
        heongpu::HEEncoder<S> encoder;
        heongpu::HEArithmeticOperator<S> operators;
        heongpu::Secretkey<S> secret_key;
        heongpu::Publickey<S> public_key;
        heongpu::Relinkey<S> relin_key;
        heongpu::Galoiskey<S> galois_key;
    };

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

    template <typename T> class SerializationContext : public ::testing::Test
    {
      protected:
        static constexpr heongpu::Scheme S = T::value;
        Party<S> a{make_context<S>(0)};
        Party<S> b{make_context<S>(1)};
    };

    template <heongpu::Scheme V>
    using SchemeValue = std::integral_constant<heongpu::Scheme, V>;
    using Schemes = ::testing::Types<SchemeValue<heongpu::Scheme::CKKS>,
                                     SchemeValue<heongpu::Scheme::BFV>>;
} // namespace

TYPED_TEST_SUITE(SerializationContext, Schemes);

TYPED_TEST(SerializationContext, Ciphertext_Loaded_For_Another_Context)
{
    constexpr auto S = TestFixture::S;
    const std::string binary = save(this->a.encrypt(1));

    heongpu::Ciphertext<S> bound(this->b.context);
    EXPECT_THROW(load(bound, binary), std::invalid_argument);

    heongpu::Ciphertext<S> loaded;
    load(loaded, binary);
    auto ciphertext_b = this->b.encrypt(1);
    heongpu::Ciphertext<S> output(this->b.context);
    EXPECT_THROW(this->b.operators.add(loaded, ciphertext_b, output),
                 std::invalid_argument);

    heongpu::HEDecryptor<S> decryptor(this->b.context, this->b.secret_key);
    heongpu::Plaintext<S> plaintext(this->b.context);
    EXPECT_THROW(decryptor.decrypt(plaintext, loaded), std::invalid_argument);
}

TYPED_TEST(SerializationContext, Plaintext_Loaded_For_Another_Context)
{
    constexpr auto S = TestFixture::S;
    const std::string binary = save(this->a.encode(1));

    heongpu::Plaintext<S> bound(this->b.context);
    EXPECT_THROW(load(bound, binary), std::invalid_argument);

    heongpu::Plaintext<S> loaded;
    load(loaded, binary);
    auto ciphertext_b = this->b.encrypt(1);
    heongpu::Ciphertext<S> output(this->b.context);
    EXPECT_THROW(this->b.operators.add_plain(ciphertext_b, loaded, output),
                 std::invalid_argument);
}

TYPED_TEST(SerializationContext, Keys_Loaded_For_Another_Context)
{
    constexpr auto S = TestFixture::S;
    auto ciphertext_b = this->b.encrypt(1);
    heongpu::Ciphertext<S> product(this->b.context);
    this->b.operators.multiply(ciphertext_b, ciphertext_b, product);

    heongpu::Relinkey<S> relin_key;
    load(relin_key, save(this->a.relin_key));
    EXPECT_THROW(this->b.operators.relinearize_inplace(product, relin_key),
                 std::invalid_argument);

    heongpu::Galoiskey<S> galois_key;
    load(galois_key, save(this->a.galois_key));
    heongpu::Ciphertext<S> rotated(this->b.context);
    EXPECT_THROW(
        rotate(this->b.operators, ciphertext_b, rotated, galois_key, 1),
        std::invalid_argument);

    heongpu::Publickey<S> public_key;
    load(public_key, save(this->a.public_key));
    EXPECT_THROW(heongpu::HEEncryptor<S>(this->b.context, public_key),
                 std::invalid_argument);

    heongpu::Secretkey<S> secret_key;
    load(secret_key, save(this->a.secret_key));
    EXPECT_THROW(heongpu::HEDecryptor<S>(this->b.context, secret_key),
                 std::invalid_argument);
}

TYPED_TEST(SerializationContext,
           Objects_Load_For_A_Context_With_The_Same_Parameters)
{
    constexpr auto S = TestFixture::S;
    Party<S> a2{make_context<S>(0)};
    const std::string binary = save(this->a.encrypt(1));

    heongpu::Ciphertext<S> bound(a2.context);
    EXPECT_NO_THROW(load(bound, binary));

    heongpu::Relinkey<S> relin_key;
    load(relin_key, save(this->a.relin_key));
    auto ciphertext = a2.encrypt(1);
    heongpu::Ciphertext<S> product(a2.context);
    a2.operators.multiply(ciphertext, ciphertext, product);
    EXPECT_NO_THROW(a2.operators.relinearize_inplace(product, relin_key));

    heongpu::Secretkey<S> secret_key;
    load(secret_key, save(this->a.secret_key));
    EXPECT_NO_THROW(heongpu::HEDecryptor<S>(a2.context, secret_key));
}

TYPED_TEST(SerializationContext, Loaded_Context_Has_The_Same_Identity)
{
    constexpr auto S = TestFixture::S;
    auto loaded = heongpu::GenHEContext<S>();
    load(*loaded, save(*this->a.context));
    EXPECT_EQ(loaded->context_id(), this->a.context->context_id());
}
