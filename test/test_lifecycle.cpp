// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <cmath>
#include <vector>

// heongpu::initialize() and heongpu::shutdown() set up and release the
// process-wide state, so that it can be set up again with another
// configuration.

namespace
{
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr size_t mebibyte = size_t(1) << 20;

    heongpu::HEContext<scheme> make_context()
    {
        auto context =
            heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
        context->generate();
        return context;
    }

    // Encrypts, adds and decrypts, which needs the memory pool and the random
    // number generator.
    bool encryption_works()
    {
        auto context = make_context();
        heongpu::HEKeyGenerator<scheme> generator(context);
        heongpu::Secretkey<scheme> secret_key(context);
        generator.generate_secret_key(secret_key);
        heongpu::Publickey<scheme> public_key(context);
        generator.generate_public_key(public_key, secret_key);
        heongpu::HEEncoder<scheme> encoder(context);
        heongpu::HEEncryptor<scheme> encryptor(context, public_key);
        heongpu::HEDecryptor<scheme> decryptor(context, secret_key);
        heongpu::HEArithmeticOperator<scheme> operators(context, encoder);

        heongpu::Plaintext<scheme> plaintext(context);
        encoder.encode(plaintext, std::vector<double>(2048, 0.5), pow(2.0, 30));
        heongpu::Ciphertext<scheme> ciphertext(context);
        encryptor.encrypt(ciphertext, plaintext);
        operators.add_inplace(ciphertext, ciphertext);
        decryptor.decrypt(plaintext, ciphertext);
        std::vector<double> result;
        encoder.decode(result, plaintext);
        return std::fabs(result[0] - 1.0) < 1e-3;
    }

    heongpu::MemoryPoolConfig config_with_pool_size(size_t bytes)
    {
        heongpu::MemoryPoolConfig config;
        config.initial_device_bytes = bytes;
        return config;
    }
} // namespace

TEST(HEonGPU, Shutdown_Is_Refused_While_Objects_Use_The_Pool)
{
    heongpu::initialize();
    {
        auto context = make_context();
        heongpu::Ciphertext<scheme> ciphertext(context);
        EXPECT_THROW(heongpu::shutdown(), heongpu::StateError);
        // Nothing was released.
        EXPECT_NE(heongpu::MemoryPool::instance().get_free_device_pool_memory(),
                  0u);
    }
    EXPECT_NO_THROW(heongpu::shutdown());
    EXPECT_EQ(heongpu::MemoryPool::instance().get_free_device_pool_memory(),
              0u);
}

TEST(HEonGPU, Restart_With_Another_Configuration)
{
    auto& pool = heongpu::MemoryPool::instance();
    heongpu::shutdown();

    heongpu::initialize(config_with_pool_size(64 * mebibyte));
    EXPECT_EQ(pool.get_free_device_pool_memory(), 64 * mebibyte);
    EXPECT_TRUE(encryption_works());
    heongpu::shutdown();

    heongpu::initialize(config_with_pool_size(128 * mebibyte));
    EXPECT_EQ(pool.get_free_device_pool_memory(), 128 * mebibyte);
    EXPECT_TRUE(encryption_works());
    heongpu::shutdown();

    // generate() sets HEonGPU up again on its own.
    EXPECT_TRUE(encryption_works());
}

TEST(HEonGPU, Shutdown_Restores_The_RMM_Default)
{
    heongpu::shutdown();
    heongpu::MemoryPoolConfig config;
    config.set_as_rmm_default = true;
    heongpu::initialize(config);
    heongpu::shutdown();
    // The RMM default must not refer to the destroyed pool.
    EXPECT_NO_THROW(rmm::device_buffer(mebibyte, rmm::cuda_stream_default));
}

TEST(HEonGPU, Shutdown_Without_Initialization_Does_Nothing)
{
    EXPECT_NO_THROW(heongpu::shutdown());
    EXPECT_NO_THROW(heongpu::shutdown());
}
