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

namespace
{
    constexpr int degree = 4096;

    std::string hex(uint64_t value)
    {
        std::ostringstream stream;
        stream << std::hex << value;
        return stream.str();
    }

    template <typename T> std::string streamed(const T& object)
    {
        std::ostringstream stream;
        stream << object;
        return stream.str();
    }
} // namespace

TEST(HEonGPU, CKKS_To_String)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    EXPECT_EQ(context->to_string(), "HEContext<CKKS>(not generated)");
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();

    const std::string context_text =
        "HEContext<CKKS>(N=4096, Q=40+30+30 bits, P=40 bits, "
        "KEYSWITCHING_METHOD_II, security=none, id=0x" +
        hex(context->context_id()) + ")";
    EXPECT_EQ(context->to_string(), context_text);
    EXPECT_EQ(streamed(*context), context_text);
    EXPECT_EQ(streamed(context), context_text);
    EXPECT_EQ(streamed(heongpu::HEContext<S>()), "HEContext<CKKS>(null)");

    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<S> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEArithmeticOperator<S> operators(context, encoder);

    heongpu::Plaintext<S> plaintext(context);
    EXPECT_EQ(plaintext.to_string(), "Plaintext<CKKS>(not generated)");
    encoder.encode(plaintext, std::vector<double>(degree / 2, 0.5),
                   pow(2.0, 30));
    EXPECT_EQ(plaintext.to_string(),
              "Plaintext<CKKS>(size=12288, depth=0, scale=2^30, SLOT, NTT "
              "domain, DEVICE)");
    EXPECT_EQ(streamed(plaintext), plaintext.to_string());

    heongpu::Ciphertext<S> ciphertext;
    EXPECT_EQ(ciphertext.to_string(), "Ciphertext<CKKS>(not generated)");
    encryptor.encrypt(ciphertext, plaintext);
    EXPECT_EQ(ciphertext.to_string(),
              "Ciphertext<CKKS>(N=4096, size=2, level=2, depth=0, "
              "scale=2^30, SLOT, NTT domain, DEVICE)");
    EXPECT_EQ(streamed(ciphertext), ciphertext.to_string());

    heongpu::Ciphertext<S> product;
    operators.multiply(ciphertext, ciphertext, product);
    EXPECT_EQ(product.to_string(),
              "Ciphertext<CKKS>(N=4096, size=3, level=2, depth=0, "
              "scale=2^60, SLOT, NTT domain, DEVICE, rescale required, "
              "relinearization required)");
}

TEST(HEonGPU, BFV_To_String)
{
    constexpr auto S = heongpu::Scheme::BFV;
    auto context = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({36, 36}, {37});
    context->set_plain_modulus(1032193);
    context->generate();

    const std::string context_text =
        "HEContext<BFV>(N=4096, Q=36+36 bits, P=37 bits, t=1032193, "
        "KEYSWITCHING_METHOD_II, security=none, id=0x" +
        hex(context->context_id()) + ")";
    EXPECT_EQ(context->to_string(), context_text);
    EXPECT_EQ(streamed(context), context_text);

    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> secret_key(context);
    generator.generate_secret_key(secret_key);
    heongpu::Publickey<S> public_key(context);
    generator.generate_public_key(public_key, secret_key);
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);

    heongpu::Plaintext<S> plaintext(context);
    encoder.encode(plaintext, std::vector<uint64_t>(degree, 3));
    EXPECT_EQ(plaintext.to_string(),
              "Plaintext<BFV>(size=4096, coefficient domain, DEVICE)");

    heongpu::Ciphertext<S> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);
    EXPECT_EQ(ciphertext.to_string(),
              "Ciphertext<BFV>(N=4096, size=2, moduli=2, coefficient domain, "
              "DEVICE)");
    EXPECT_EQ(streamed(ciphertext), ciphertext.to_string());
}

TEST(HEonGPU, Print_Parameters_Writes_To_A_Stream)
{
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
        heongpu::sec_level_type::none);
    std::ostringstream before;
    context->print_parameters(before);
    EXPECT_EQ(before.str(), "Parameters is not generated yet!\n");

    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();
    std::ostringstream stream;
    context->print_parameters(stream);
    EXPECT_EQ(stream.str(),
              "==== HEonGPU a GPU Based Homomorphic Encryption Library ====\n"
              "\n"
              "Encryption parameters:\n"
              "-->   scheme: CKKS\n"
              "-->   poly_modulus_degree: 4096\n"
              "-->   Q_tilda size: Q( 40 + 30 + 30 ) + P( 40 ) bits\n"
              "\n");
}

TEST(HEonGPU, Memory_Pool_Status_Writes_To_A_Stream)
{
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
        heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
    context->generate();
    std::ostringstream stream;
    heongpu::MemoryPool::instance().print_memory_pool_status(stream);
    EXPECT_EQ(stream.str().rfind("[HEonGPU] Memory Pool Status\n", 0), 0u)
        << stream.str();
}
