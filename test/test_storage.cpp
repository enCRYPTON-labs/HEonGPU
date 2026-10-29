// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// store_in_device keeps the data when the copy to the device runs after it
// returns: the host memory, from a pool, is released only after the copy.

namespace
{
    template <typename T> std::string bytes(const T& object)
    {
        std::stringstream stream;
        object.save(stream);
        return stream.str();
    }

    // Holds back the work queued on stream after it for a while.
    void delay(cudaStream_t stream)
    {
        ASSERT_EQ(cudaSuccess, cudaLaunchHostFunc(
                                   stream,
                                   [](void*) {
                                       std::this_thread::sleep_for(
                                           std::chrono::milliseconds(200));
                                   },
                                   nullptr));
    }

    // Moves object to the device on a delayed stream and fills host memory
    // of the pool, which reuses the memory object released, before the copy
    // runs.
    template <typename T> void expect_kept(T& object, std::size_t size)
    {
        const std::string before = bytes(object);
        object.store_in_host();
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        // The memory object moves to stays bound to the stream, which lives
        // as long as the process.
        static cudaStream_t stream = []
        {
            cudaStream_t created;
            cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
            return created;
        }();
        delay(stream);
        object.store_in_device(stream);
        std::vector<heongpu::HostVector<Data64>> junk;
        for (int i = 0; i < 4; i++)
        {
            junk.emplace_back(size, ~Data64(0));
        }
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_TRUE(object.is_on_device());
        const std::string after = bytes(object);
        std::size_t first = 0;
        while (first < before.size() && first < after.size() &&
               before[first] == after[first])
        {
            first++;
        }
        EXPECT_EQ(first, before.size())
            << "sizes " << before.size() << " " << after.size();
    }
} // namespace

TEST(Storage, CKKS_Store_In_Device_Keeps_The_Data)
{
    constexpr auto S = heongpu::Scheme::CKKS;
    heongpu::HEParams<S> params;
    params.multiplicative_depth = 3;
    auto context = params.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    auto relin_key = keygen.relin_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    const std::size_t n = context->get_poly_modulus_degree();

    auto plain =
        encoder.encode(std::vector<double>(n / 2, 0.5), params.scale());
    expect_kept(plain, plain.size());
    expect_kept(public_key, 2 * n * context->get_key_modulus().size());
    expect_kept(relin_key, 2 * n * context->get_key_modulus().size());
}

TEST(Storage, BFV_Store_In_Device_Keeps_The_Data)
{
    constexpr auto S = heongpu::Scheme::BFV;
    heongpu::HEParams<S> params;
    params.multiplicative_depth = 2;
    params.plain_modulus_bit_size = 20;
    auto context = params.make_context();
    context->generate();
    heongpu::HEKeyGenerator<S> keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key(secret_key);
    heongpu::HEEncoder<S> encoder(context);
    const std::size_t n = context->get_poly_modulus_degree();

    auto plain = encoder.encode(std::vector<uint64_t>(n, 3));
    expect_kept(plain, plain.size());
    expect_kept(public_key, 2 * n * context->get_key_modulus().size());
}
