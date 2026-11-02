// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

// The collective relinearization key of the multiparty computation
// relinearizes products correctly. Every error polynomial of the round 2
// share is sampled: the memory pool is filled with nonzero bytes before the
// shares are generated, so a polynomial that is read without being sampled
// gives wrong results.

namespace
{
    constexpr int parties = 3;

    void dirty_memory_pool()
    {
        const size_t bytes = size_t(64) << 20;
        heongpu::DeviceVector<Data64> memory(bytes / sizeof(Data64));
        ASSERT_EQ(cudaMemset(memory.data(), 0x5a, bytes), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    template <heongpu::Scheme S> struct Parties
    {
        std::vector<std::unique_ptr<heongpu::Secretkey<S>>> secret_keys;
        std::vector<std::unique_ptr<heongpu::HEMultiPartyManager<S>>> managers;
    };

    // Generates the collective public and relinearization keys of the
    // parties; the manager of the first party also assembles the shares.
    template <heongpu::Scheme S>
    void collective_keys(heongpu::HEContext<S>& context, Parties<S>& parties,
                         heongpu::Publickey<S>& public_key,
                         heongpu::Relinkey<S>& relin_key)
    {
        heongpu::RNGSeed common_seed;
        auto& server = *parties.managers[0];

        std::vector<heongpu::MultipartyPublickey<S>> public_key_shares;
        std::vector<heongpu::MultipartyRelinkey<S>> relin_key_inits;
        for (int i = 0; i < ::parties; i++)
        {
            heongpu::MultipartyPublickey<S> public_key_share(context,
                                                             common_seed);
            parties.managers[i]->generate_public_key_share(
                public_key_share, *parties.secret_keys[i]);
            public_key_shares.push_back(public_key_share);

            heongpu::MultipartyRelinkey<S> relin_key_init(context, common_seed);
            parties.managers[i]->generate_relin_key_init(
                relin_key_init, *parties.secret_keys[i]);
            relin_key_inits.push_back(relin_key_init);
        }
        server.assemble_public_key_share(public_key_shares, public_key);

        heongpu::MultipartyRelinkey<S> common_relin_key_init(context,
                                                             common_seed);
        server.assemble_relin_key_init(relin_key_inits, common_relin_key_init);

        std::vector<heongpu::MultipartyRelinkey<S>> relin_key_shares;
        for (int i = 0; i < ::parties; i++)
        {
            dirty_memory_pool();
            heongpu::MultipartyRelinkey<S> relin_key_share(context,
                                                           common_seed);
            parties.managers[i]->generate_relin_key_share(
                common_relin_key_init, relin_key_share,
                *parties.secret_keys[i]);
            relin_key_shares.push_back(relin_key_share);
        }
        server.assemble_relin_key_share(relin_key_shares, common_relin_key_init,
                                        relin_key);
    }

    template <heongpu::Scheme S>
    void decrypt(heongpu::HEContext<S>& context, Parties<S>& parties,
                 heongpu::Ciphertext<S>& ciphertext,
                 heongpu::Plaintext<S>& plaintext)
    {
        std::vector<heongpu::Ciphertext<S>> partial_ciphertexts;
        for (int i = 0; i < ::parties; i++)
        {
            heongpu::Ciphertext<S> partial_ciphertext(context);
            parties.managers[i]->decrypt_partial(
                ciphertext, *parties.secret_keys[i], partial_ciphertext);
            partial_ciphertexts.push_back(partial_ciphertext);
        }
        parties.managers[0]->decrypt(partial_ciphertexts, plaintext);
    }
} // namespace

TEST(MpcRelinkey, CkksCollectiveKeyRelinearizesProducts)
{
    constexpr heongpu::Scheme S = heongpu::Scheme::CKKS;
    heongpu::HEParams<S> params;
    params.security_level = heongpu::sec_level_type::none;
    params.poly_modulus_degree = 8192;
    params.coeff_modulus_bit_sizes = {60, 50, 50, 50};
    params.special_modulus_bit_sizes = {60};
    heongpu::HEContext<S> context = params.make_context();
    context->generate();
    double scale = std::pow(2.0, 50);
    const int slot_count = params.poly_modulus_degree / 2;

    heongpu::HEEncoder<S> encoder(context);
    Parties<S> parties;
    for (int i = 0; i < ::parties; i++)
    {
        heongpu::HEKeyGenerator<S> keygen(context);
        parties.secret_keys.push_back(
            std::make_unique<heongpu::Secretkey<S>>(context));
        keygen.generate_secret_key(*parties.secret_keys[i]);
        parties.managers.push_back(
            std::make_unique<heongpu::HEMultiPartyManager<S>>(context, encoder,
                                                              scale));
    }

    heongpu::Publickey<S> public_key(context);
    heongpu::Relinkey<S> relin_key(context);
    collective_keys(context, parties, public_key, relin_key);

    std::vector<double> x(slot_count), y(slot_count), expected(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        x[i] = std::sin(0.01 * i);
        y[i] = std::cos(0.02 * i);
        expected[i] = x[i] * y[i];
    }

    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::Plaintext<S> plaintext_x(context), plaintext_y(context);
    encoder.encode(plaintext_x, x, scale);
    encoder.encode(plaintext_y, y, scale);
    heongpu::Ciphertext<S> ciphertext_x(context), ciphertext_y(context);
    encryptor.encrypt(ciphertext_x, plaintext_x);
    encryptor.encrypt(ciphertext_y, plaintext_y);

    heongpu::HEArithmeticOperator<S> operators(context, encoder);
    heongpu::Ciphertext<S> product(context);
    operators.multiply(ciphertext_x, ciphertext_y, product);
    operators.relinearize_inplace(product, relin_key);
    operators.rescale_inplace(product);

    heongpu::Plaintext<S> plaintext(context);
    decrypt(context, parties, product, plaintext);
    std::vector<double> result;
    encoder.decode(result, plaintext);

    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < slot_count; i++)
    {
        ASSERT_NEAR(result[i], expected[i], 1e-3) << "slot " << i;
    }
}

TEST(MpcRelinkey, BfvCollectiveKeyRelinearizesProducts)
{
    constexpr heongpu::Scheme S = heongpu::Scheme::BFV;
    heongpu::HEParams<S> params;
    params.poly_modulus_degree = 8192;
    params.coeff_modulus_bit_sizes = {43, 43, 44, 44};
    params.special_modulus_bit_sizes = {44};
    params.plain_modulus = 1032193;
    heongpu::HEContext<S> context = params.make_context();
    context->generate();
    const uint64_t t = params.plain_modulus;
    const int slot_count = params.poly_modulus_degree;

    Parties<S> parties;
    for (int i = 0; i < ::parties; i++)
    {
        heongpu::HEKeyGenerator<S> keygen(context);
        parties.secret_keys.push_back(
            std::make_unique<heongpu::Secretkey<S>>(context));
        keygen.generate_secret_key(*parties.secret_keys[i]);
        parties.managers.push_back(
            std::make_unique<heongpu::HEMultiPartyManager<S>>(context));
    }

    heongpu::Publickey<S> public_key(context);
    heongpu::Relinkey<S> relin_key(context);
    collective_keys(context, parties, public_key, relin_key);

    std::vector<uint64_t> x(slot_count), y(slot_count), expected(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        x[i] = (uint64_t(i) * 7919) % t;
        y[i] = (uint64_t(i) * 104729 + 3) % t;
        expected[i] = (x[i] * y[i]) % t;
    }

    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::Plaintext<S> plaintext_x(context), plaintext_y(context);
    encoder.encode(plaintext_x, x);
    encoder.encode(plaintext_y, y);
    heongpu::Ciphertext<S> ciphertext_x(context), ciphertext_y(context);
    encryptor.encrypt(ciphertext_x, plaintext_x);
    encryptor.encrypt(ciphertext_y, plaintext_y);

    heongpu::HEArithmeticOperator<S> operators(context, encoder);
    heongpu::Ciphertext<S> product(context);
    operators.multiply(ciphertext_x, ciphertext_y, product);
    operators.relinearize_inplace(product, relin_key);

    heongpu::Plaintext<S> plaintext(context);
    decrypt(context, parties, product, plaintext);
    std::vector<uint64_t> result;
    encoder.decode(result, plaintext);

    ASSERT_EQ(result, expected);
}
