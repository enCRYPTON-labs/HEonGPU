// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// The random number generator gives the same numbers for a seed on every
// kind of stream, which seeded generation, as of the common polynomials of
// multiparty computation, needs.

namespace
{
    using heongpu::CudaException;

    constexpr int log_size = 16;
    constexpr int mod_count = 8;
    constexpr int requests = 4;

    std::vector<Data64> generate(cudaStream_t stream, Modulus64* modulus)
    {
        auto& generator = heongpu::RandomNumberGenerator::instance();
        std::vector<unsigned char> key(16, 7), nonce(8, 3), personalization;
        generator.set(key, nonce, personalization, stream);

        std::size_t size = (std::size_t(1) << log_size) * mod_count;
        heongpu::DeviceVector<Data64> values(size);
        std::vector<Data64> all;
        for (int i = 0; i < requests; i++)
        {
            generator.modular_uniform_random_number_generation(
                values.data(), modulus, log_size, mod_count, 1, stream);
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<Data64> host(size);
            HEONGPU_CUDA_CHECK(cudaMemcpy(host.data(), values.data(),
                                          size * sizeof(Data64),
                                          cudaMemcpyDeviceToHost));
            all.insert(all.end(), host.begin(), host.end());
        }
        return all;
    }
} // namespace

TEST(HEonGPU, Random_Numbers_Of_A_Seed_Do_Not_Depend_On_The_Stream)
{
    // A generated context initializes the generator.
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>();
    context->set_poly_modulus_degree(1 << log_size);
    context->set_coeff_modulus_bit_sizes(std::vector<int>(mod_count - 1, 50),
                                         {60});
    context->generate();

    cudaStream_t blocking, non_blocking;
    HEONGPU_CUDA_CHECK(cudaStreamCreate(&blocking));
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&non_blocking, cudaStreamNonBlocking));

    heongpu::DeviceVector<Modulus64> modulus(context->get_key_modulus());
    std::vector<Data64> reference = generate(cudaStreamDefault, modulus.data());
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(generate(blocking, modulus.data()), reference);
        EXPECT_EQ(generate(non_blocking, modulus.data()), reference);
    }

    // Leave the generator with a fresh seed.
    heongpu::RNGSeed seed;
    heongpu::RandomNumberGenerator::instance().set(
        seed.key_, seed.nonce_, seed.personalization_string_);

    HEONGPU_CUDA_CHECK(cudaStreamDestroy(blocking));
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(non_blocking));
}

TEST(HEonGPU, With_Seed_Is_Not_Interrupted_By_Other_Threads)
{
    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>();
    context->set_poly_modulus_degree(1 << log_size);
    context->set_coeff_modulus_bit_sizes(std::vector<int>(mod_count - 1, 50),
                                         {60});
    context->generate();
    heongpu::DeviceVector<Modulus64> modulus(context->get_key_modulus());
    auto& generator = heongpu::RandomNumberGenerator::instance();
    std::size_t size = (std::size_t(1) << log_size) * mod_count;

    // Two requests with the seed: the first is compared, the second is what
    // the generator would give next without a new seed.
    heongpu::RNGSeed seed(std::vector<unsigned char>(16, 9),
                          std::vector<unsigned char>(8, 4), {});
    auto seeded = [&](cudaStream_t stream)
    {
        heongpu::DeviceVector<Data64> values(2 * size);
        generator.with_seed(
            seed, stream,
            [&]
            {
                // Gives other threads time to draw numbers in between, which
                // with_seed must prevent.
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                for (int i = 0; i < 2; i++)
                {
                    generator.modular_uniform_random_number_generation(
                        values.data() + i * size, modulus.data(), log_size,
                        mod_count, 1, stream);
                }
            });
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<Data64> host(2 * size);
        HEONGPU_CUDA_CHECK(cudaMemcpy(host.data(), values.data(),
                                      2 * size * sizeof(Data64),
                                      cudaMemcpyDeviceToHost));
        return host;
    };
    std::vector<Data64> reference = seeded(cudaStreamDefault);

    // After with_seed the generator has a new seed.
    heongpu::DeviceVector<Data64> next(size);
    generator.modular_uniform_random_number_generation(
        next.data(), modulus.data(), log_size, mod_count, 1);
    std::vector<Data64> next_host(size);
    HEONGPU_CUDA_CHECK(cudaMemcpy(next_host.data(), next.data(),
                                  size * sizeof(Data64),
                                  cudaMemcpyDeviceToHost));
    EXPECT_FALSE(std::equal(next_host.begin(), next_host.end(),
                            reference.begin() + size));

    // Other threads draw numbers meanwhile.
    std::atomic<bool> done(false);
    std::vector<std::thread> others;
    for (int t = 0; t < 3; t++)
    {
        others.emplace_back(
            [&]
            {
                cudaStream_t stream;
                HEONGPU_CUDA_CHECK(
                    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                {
                    heongpu::DeviceVector<Data64> values(size, stream);
                    while (!done)
                    {
                        generator.modular_uniform_random_number_generation(
                            values.data(), modulus.data(), log_size, mod_count,
                            1, stream);
                    }
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
            });
    }
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < 10; i++)
    {
        EXPECT_EQ(seeded(stream), reference);
    }
    done = true;
    for (auto& thread : others)
    {
        thread.join();
    }
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}
