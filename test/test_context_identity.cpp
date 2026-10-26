// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <vector>

// Objects of two contexts with the same ring size and the same number of
// moduli, but different primes, must not be mixed.

namespace
{
    constexpr int degree = 4096;

    template <heongpu::Scheme S> struct Party
    {
        explicit Party(heongpu::HEContext<S> context_)
            : context(context_), encoder(context)
        {
            heongpu::HEKeyGenerator<S> generator(context);
            secret_key = heongpu::Secretkey<S>(context);
            generator.generate_secret_key(secret_key);
            public_key = heongpu::Publickey<S>(context);
            generator.generate_public_key(public_key, secret_key);
            relin_key = heongpu::Relinkey<S>(context);
            generator.generate_relin_key(relin_key, secret_key);
        }

        heongpu::HEContext<S> context;
        heongpu::HEEncoder<S> encoder;
        heongpu::Secretkey<S> secret_key;
        heongpu::Publickey<S> public_key;
        heongpu::Relinkey<S> relin_key;
    };

    using CKKSParty = Party<heongpu::Scheme::CKKS>;
    using BFVParty = Party<heongpu::Scheme::BFV>;

    heongpu::HEContext<heongpu::Scheme::CKKS>
    make_ckks_context(const std::vector<int>& q_bits)
    {
        auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(degree);
        context->set_coeff_modulus_bit_sizes(q_bits, {40});
        context->generate();
        return context;
    }

    heongpu::HEContext<heongpu::Scheme::BFV>
    make_bfv_context(const std::vector<int>& q_bits)
    {
        auto context = heongpu::GenHEContext<heongpu::Scheme::BFV>(
            heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(degree);
        context->set_coeff_modulus_bit_sizes(q_bits, {37});
        context->set_plain_modulus(1032193);
        context->generate();
        return context;
    }

    heongpu::Plaintext<heongpu::Scheme::CKKS> encode(CKKSParty& party,
                                                     double value)
    {
        heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext(party.context);
        party.encoder.encode(plaintext, std::vector<double>(degree / 2, value),
                             pow(2.0, 30));
        return plaintext;
    }

    heongpu::Ciphertext<heongpu::Scheme::CKKS> encrypt(CKKSParty& party,
                                                       double value)
    {
        auto plaintext = encode(party, value);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(party.context,
                                                              party.public_key);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> ciphertext(party.context);
        encryptor.encrypt(ciphertext, plaintext);
        return ciphertext;
    }

    heongpu::Plaintext<heongpu::Scheme::BFV> encode(BFVParty& party,
                                                    uint64_t value)
    {
        heongpu::Plaintext<heongpu::Scheme::BFV> plaintext(party.context);
        party.encoder.encode(plaintext, std::vector<uint64_t>(degree, value));
        return plaintext;
    }

    heongpu::Ciphertext<heongpu::Scheme::BFV> encrypt(BFVParty& party,
                                                      uint64_t value)
    {
        auto plaintext = encode(party, value);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(party.context,
                                                             party.public_key);
        heongpu::Ciphertext<heongpu::Scheme::BFV> ciphertext(party.context);
        encryptor.encrypt(ciphertext, plaintext);
        return ciphertext;
    }

    class CKKSContextIdentity : public ::testing::Test
    {
      protected:
        CKKSParty a{make_ckks_context({40, 30, 30})};
        CKKSParty b{make_ckks_context({40, 31, 31})};
    };

    class BFVContextIdentity : public ::testing::Test
    {
      protected:
        BFVParty a{make_bfv_context({36, 36})};
        BFVParty b{make_bfv_context({35, 35})};
    };
} // namespace

TEST_F(CKKSContextIdentity, Ciphertext_Of_Another_Context_Is_Rejected)
{
    auto ciphertext_a = encrypt(a, 0.25);
    auto ciphertext_b = encrypt(b, 0.25);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_b(b.context,
                                                                     b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output(b.context);

    EXPECT_THROW(operators_b.add(ciphertext_a, ciphertext_b, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.add(ciphertext_b, ciphertext_a, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.multiply(ciphertext_a, ciphertext_b, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.negate(ciphertext_a, output),
                 std::invalid_argument);
    EXPECT_NO_THROW(operators_b.add(ciphertext_b, ciphertext_b, output));
}

TEST_F(CKKSContextIdentity, Plaintext_Of_Another_Context_Is_Rejected)
{
    auto ciphertext_b = encrypt(b, 0.25);
    auto plaintext_a = encode(a, 0.5);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_b(b.context,
                                                                     b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output(b.context);

    EXPECT_THROW(operators_b.add_plain(ciphertext_b, plaintext_a, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.multiply_plain(ciphertext_b, plaintext_a, output),
                 std::invalid_argument);

    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_b(b.context,
                                                            b.public_key);
    EXPECT_THROW(encryptor_b.encrypt(output, plaintext_a),
                 std::invalid_argument);

    std::vector<double> decoded;
    EXPECT_THROW(b.encoder.decode(decoded, plaintext_a), std::invalid_argument);
}

TEST_F(CKKSContextIdentity, Decryption_With_Another_Context_Is_Rejected)
{
    auto ciphertext_a = encrypt(a, 0.25);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor_b(b.context,
                                                            b.secret_key);
    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext(b.context);
    EXPECT_THROW(decryptor_b.decrypt(plaintext, ciphertext_a),
                 std::invalid_argument);
}

TEST_F(CKKSContextIdentity, Keys_Of_Another_Context_Are_Rejected)
{
    auto ciphertext_b = encrypt(b, 0.25);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_b(b.context,
                                                                     b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> product(b.context);
    operators_b.multiply(ciphertext_b, ciphertext_b, product);
    EXPECT_THROW(operators_b.relinearize_inplace(product, a.relin_key),
                 std::invalid_argument);

    EXPECT_THROW(
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>(b.context, a.public_key),
        std::invalid_argument);
}

TEST_F(CKKSContextIdentity, Same_Parameters_Give_Compatible_Contexts)
{
    // A context generated again from the same parameters (for example on
    // another machine) works with the objects of the first one.
    CKKSParty a2{make_ckks_context({40, 30, 30})};
    auto ciphertext_a = encrypt(a, 0.25);
    auto ciphertext_a2 = encrypt(a2, 0.5);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_a2(
        a2.context, a2.encoder);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output(a2.context);
    EXPECT_NO_THROW(operators_a2.add(ciphertext_a, ciphertext_a2, output));
}

TEST_F(BFVContextIdentity, Ciphertext_Of_Another_Context_Is_Rejected)
{
    auto ciphertext_a = encrypt(a, 3);
    auto ciphertext_b = encrypt(b, 3);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_b(b.context,
                                                                    b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::BFV> output(b.context);

    EXPECT_THROW(operators_b.add(ciphertext_a, ciphertext_b, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.add(ciphertext_b, ciphertext_a, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.multiply(ciphertext_a, ciphertext_b, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.negate(ciphertext_a, output),
                 std::invalid_argument);
    EXPECT_NO_THROW(operators_b.add(ciphertext_b, ciphertext_b, output));
}

TEST_F(BFVContextIdentity, Plaintext_Of_Another_Context_Is_Rejected)
{
    auto ciphertext_b = encrypt(b, 3);
    auto plaintext_a = encode(a, 5);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_b(b.context,
                                                                    b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::BFV> output(b.context);

    EXPECT_THROW(operators_b.add_plain(ciphertext_b, plaintext_a, output),
                 std::invalid_argument);
    EXPECT_THROW(operators_b.multiply_plain(ciphertext_b, plaintext_a, output),
                 std::invalid_argument);

    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor_b(b.context,
                                                           b.public_key);
    EXPECT_THROW(encryptor_b.encrypt(output, plaintext_a),
                 std::invalid_argument);
}

TEST_F(BFVContextIdentity, Decryption_With_Another_Context_Is_Rejected)
{
    auto ciphertext_a = encrypt(a, 3);
    heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor_b(b.context,
                                                           b.secret_key);
    heongpu::Plaintext<heongpu::Scheme::BFV> plaintext(b.context);
    EXPECT_THROW(decryptor_b.decrypt(plaintext, ciphertext_a),
                 std::invalid_argument);
}

TEST_F(BFVContextIdentity, Keys_Of_Another_Context_Are_Rejected)
{
    auto ciphertext_b = encrypt(b, 3);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_b(b.context,
                                                                    b.encoder);
    heongpu::Ciphertext<heongpu::Scheme::BFV> product(b.context);
    operators_b.multiply(ciphertext_b, ciphertext_b, product);
    EXPECT_THROW(operators_b.relinearize_inplace(product, a.relin_key),
                 std::invalid_argument);

    EXPECT_THROW(
        heongpu::HEEncryptor<heongpu::Scheme::BFV>(b.context, a.public_key),
        std::invalid_argument);
}

TEST_F(BFVContextIdentity, Same_Parameters_Give_Compatible_Contexts)
{
    BFVParty a2{make_bfv_context({36, 36})};
    auto ciphertext_a = encrypt(a, 3);
    auto ciphertext_a2 = encrypt(a2, 4);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_a2(
        a2.context, a2.encoder);
    heongpu::Ciphertext<heongpu::Scheme::BFV> output(a2.context);
    EXPECT_NO_THROW(operators_a2.add(ciphertext_a, ciphertext_a2, output));
}

TEST_F(CKKSContextIdentity, Context_Id_Depends_On_The_Primes)
{
    EXPECT_NE(a.context->context_id(), 0u);
    EXPECT_NE(a.context->context_id(), b.context->context_id());
    EXPECT_EQ(a.context->context_id(),
              make_ckks_context({40, 30, 30})->context_id());
}

TEST_F(CKKSContextIdentity, Operation_Results_Keep_The_Context)
{
    auto ciphertext_a = encrypt(a, 0.25);
    auto ciphertext_b = encrypt(b, 0.25);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_a(a.context,
                                                                     a.encoder);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_b(b.context,
                                                                     b.encoder);

    // A default constructed output takes the context of the operation.
    heongpu::Ciphertext<heongpu::Scheme::CKKS> sum_b;
    operators_b.add(ciphertext_b, ciphertext_b, sum_b);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output;
    EXPECT_THROW(operators_a.add(ciphertext_a, sum_b, output),
                 heongpu::ContextMismatchError);

    auto copy_b = sum_b;
    EXPECT_THROW(operators_a.add(ciphertext_a, copy_b, output),
                 heongpu::ContextMismatchError);
}

TEST_F(BFVContextIdentity, Operation_Results_Keep_The_Context)
{
    auto ciphertext_a = encrypt(a, 3);
    auto ciphertext_b = encrypt(b, 3);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_a(a.context,
                                                                    a.encoder);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators_b(b.context,
                                                                    b.encoder);

    heongpu::Ciphertext<heongpu::Scheme::BFV> product_b;
    operators_b.multiply(ciphertext_b, ciphertext_b, product_b);
    operators_b.relinearize_inplace(product_b, b.relin_key);
    heongpu::Ciphertext<heongpu::Scheme::BFV> output;
    EXPECT_THROW(operators_a.add(ciphertext_a, product_b, output),
                 heongpu::ContextMismatchError);
}
