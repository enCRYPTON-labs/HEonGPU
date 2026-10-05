// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <heongpu/kernel/decryption.cuh>
#include <gtest/gtest.h>
#include <gmp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>

namespace
{
    using heongpu::CudaException;

    struct Integer
    {
        mpz_t value;
        Integer() { mpz_init(value); }
        ~Integer() { mpz_clear(value); }
        Integer(const Integer&) = delete;
        Integer& operator=(const Integer&) = delete;
    };

    std::vector<Data64> limbs(mpz_srcptr value, int count)
    {
        std::vector<Data64> result(count, 0);
        mpz_export(result.data(), nullptr, -1, sizeof(Data64), 0, 0, value);
        return result;
    }

    struct DeviceBuffer
    {
        Data64* data = nullptr;
        explicit DeviceBuffer(size_t count)
        {
            HEONGPU_CUDA_CHECK(cudaMalloc(&data, count * sizeof(Data64)));
        }
        ~DeviceBuffer() { cudaFree(data); }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        void upload(const std::vector<Data64>& values)
        {
            HEONGPU_CUDA_CHECK(cudaMemcpy(data, values.data(),
                                          values.size() * sizeof(Data64),
                                          cudaMemcpyHostToDevice));
        }
    };

    void set_coefficient(std::vector<Data64>& values, int n, int index,
                         const std::vector<Data64>& coefficient)
    {
        for (size_t j = 0; j < coefficient.size(); j++)
        {
            values[index + j * n] = coefficient[j];
        }
    }

    void check_norm(const std::vector<Data64>& values, int n_power,
                    int word_count, bool in_place)
    {
        const int n = 1 << n_power;
        ASSERT_EQ(values.size(), size_t(n * word_count));
        Integer modulus, threshold, candidate, maximum;
        mpz_setbit(modulus.value, 64 * word_count);
        mpz_sub_ui(modulus.value, modulus.value, 59);
        mpz_add_ui(threshold.value, modulus.value, 1);
        mpz_fdiv_q_2exp(threshold.value, threshold.value, 1);

        // GMP provides an independent CPU oracle for centered multiword values.
        std::vector<Data64> coefficient(word_count);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < word_count; j++)
            {
                coefficient[j] = values[i + j * n];
            }
            mpz_import(candidate.value, word_count, -1, sizeof(Data64), 0, 0,
                       coefficient.data());
            ASSERT_LT(mpz_cmp(candidate.value, modulus.value), 0);
            if (mpz_cmp(candidate.value, threshold.value) >= 0)
            {
                mpz_sub(candidate.value, modulus.value, candidate.value);
            }
            if (mpz_cmp(candidate.value, maximum.value) > 0)
            {
                mpz_set(maximum.value, candidate.value);
            }
        }

        DeviceBuffer input(values.size()), output(word_count), q(word_count),
            half(word_count);
        input.upload(values);
        q.upload(limbs(modulus.value, word_count));
        half.upload(limbs(threshold.value, word_count));
        Data64* destination = in_place ? input.data : output.data;
        heongpu::find_max_norm_kernel<<<1, 512, 512 * sizeof(Data64)>>>(
            input.data, destination, half.data, q.data, word_count, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Data64> actual(word_count);
        HEONGPU_CUDA_CHECK(cudaMemcpy(actual.data(), destination,
                                      actual.size() * sizeof(Data64),
                                      cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual, limbs(maximum.value, word_count));
    }

    constexpr auto scheme = heongpu::Scheme::BFV;
    constexpr int degree = 4096;

    class BFVNoiseBudget : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            context = heongpu::GenHEContext<scheme>();
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes({30, 30}, {30});
            context->set_plain_modulus(65537);
            heongpu::MemoryPoolConfig config;
            config.use_memory_pool = false;
            context->generate(config);
            heongpu::HEKeyGenerator<scheme> generator(context);
            heongpu::Secretkey<scheme> secret(context);
            generator.generate_secret_key(secret);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<scheme>>(context, secret);
            heongpu::Publickey<scheme> public_key(context);
            generator.generate_public_key(public_key, secret);
            heongpu::HEEncoder<scheme> encoder(context);
            heongpu::HEEncryptor<scheme> encryptor(context, public_key);
            heongpu::Plaintext<scheme> plain(context);
            encoder.encode(plain, std::vector<uint64_t>(degree, 0));
            ciphertext = std::make_unique<heongpu::Ciphertext<scheme>>(context);
            encryptor.encrypt(*ciphertext, plain);
        }

        void set_phase(Data64 value, int index, bool negative = false)
        {
            const int count = context->get_ciphertext_modulus_count();
            std::vector<Data64> values(2 * degree * count, 0);
            const auto primes = context->get_key_modulus();
            for (int j = 0; j < count; j++)
            {
                const Data64 residue = value % primes[j].value;
                values[index + j * degree] =
                    negative && residue ? primes[j].value - residue : residue;
            }
            // c1 = 0 makes the noise phase independent of the generated key.
            HEONGPU_CUDA_CHECK(cudaMemcpy(ciphertext->data(), values.data(),
                                          values.size() * sizeof(Data64),
                                          cudaMemcpyHostToDevice));
        }

        int expected_budget(Data64 value)
        {
            Integer modulus, noise;
            mpz_set_ui(modulus.value, 1);
            const auto primes = context->get_key_modulus();
            for (int i = 0; i < context->get_ciphertext_modulus_count(); i++)
            {
                mpz_mul_ui(modulus.value, modulus.value, primes[i].value);
            }
            mpz_set_ui(noise.value, value);
            mpz_mul_ui(noise.value, noise.value,
                       context->get_plain_modulus().value);
            const int bit_count =
                value == 0 ? 0 : int(mpz_sizeinbase(noise.value, 2));
            return std::max(0, int(mpz_sizeinbase(modulus.value, 2)) -
                                   bit_count - 1);
        }

        heongpu::HEContext<scheme> context;
        std::unique_ptr<heongpu::HEDecryptor<scheme>> decryptor;
        std::unique_ptr<heongpu::Ciphertext<scheme>> ciphertext;
    };
} // namespace

TEST(BFVMaxNorm, FindsMaximumAcrossReductionStagesAndPolynomialIterations)
{
    for (int n_power : {10, 12, 16})
    {
        const int n = 1 << n_power;
        for (int position : {0, 1, 2, 15, 16, 31, 32, 127, 128, 255, 256, 511,
                             512, 513, n - 1})
        {
            SCOPED_TRACE(n_power);
            SCOPED_TRACE(position);
            std::vector<Data64> values(n, 1);
            values[position] = 1000;
            check_norm(values, n_power, 1, true);
        }
    }
}

TEST(BFVMaxNorm, CentersNegativeValuesAndUsesHighLimbsForComparison)
{
    constexpr int n = 1024;
    for (int count : {1, 2, 3, 50})
    {
        Integer q, magnitude, negative;
        mpz_setbit(q.value, 64 * count);
        mpz_sub_ui(q.value, q.value, 59);
        mpz_set_ui(magnitude.value, 1000);
        if (count > 1)
        {
            mpz_setbit(magnitude.value, 64 * (count - 1));
        }
        mpz_sub(negative.value, q.value, magnitude.value);
        std::vector<Data64> values(n * count, 0);
        set_coefficient(values, n, 511, limbs(negative.value, count));
        // A large low limb must not beat a larger high limb.
        values[1] = count == 1 ? 999 : ~Data64(0);
        for (bool in_place : {false, true})
        {
            SCOPED_TRACE(count);
            SCOPED_TRACE(in_place);
            check_norm(values, 10, count, in_place);
        }
    }
}

TEST(BFVMaxNorm, MatchesCPUReferenceForRandomMultiwordCoefficients)
{
    std::mt19937_64 random(20261005);
    for (int count : {1, 2, 3, 50})
    {
        std::vector<Data64> values(4096 * count);
        for (auto& value : values)
        {
            value = random();
        }
        check_norm(values, 12, count, true);
    }
}

TEST(BFVMaxNorm, HandlesZeroAndCenteringBoundary)
{
    for (int count : {1, 3})
    {
        std::vector<Data64> values(1024 * count, 0);
        check_norm(values, 10, count, true);
        Integer q, half, below;
        mpz_setbit(q.value, 64 * count);
        mpz_sub_ui(q.value, q.value, 59);
        mpz_add_ui(half.value, q.value, 1);
        mpz_fdiv_q_2exp(half.value, half.value, 1);
        mpz_sub_ui(below.value, half.value, 1);
        set_coefficient(values, 1024, 1, limbs(half.value, count));
        set_coefficient(values, 1024, 1023, limbs(below.value, count));
        check_norm(values, 10, count, true);
    }
}

TEST_F(BFVNoiseBudget, ReturnsExactBudgetForControlledPositiveAndNegativeNoise)
{
    for (int position : {0, 1, 31, 511, 513, degree - 1})
    {
        for (bool negative : {false, true})
        {
            set_phase(1000, position, negative);
            EXPECT_EQ(decryptor->remainder_noise_budget(*ciphertext),
                      expected_budget(1000));
        }
    }
}

TEST_F(BFVNoiseBudget, HandlesZeroAndExhaustedNoiseBudget)
{
    set_phase(0, 1);
    EXPECT_EQ(decryptor->remainder_noise_budget(*ciphertext),
              expected_budget(0));
    const auto primes = context->get_key_modulus();
    const Data64 q = primes[0].value * primes[1].value;
    const Data64 phase = (q / 2) / context->get_plain_modulus().value;
    set_phase(phase, 1);
    EXPECT_EQ(decryptor->remainder_noise_budget(*ciphertext), 0);
}

TEST_F(BFVNoiseBudget, WaitsForNonblockingStreamBeforeReturningBudget)
{
    set_phase(1000, 1);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::atomic<bool> started{false};
    HEONGPU_CUDA_CHECK(cudaLaunchHostFunc(
        stream,
        [](void* pointer)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            static_cast<std::atomic<bool>*>(pointer)->store(true);
        },
        &started));
    EXPECT_EQ(decryptor->remainder_noise_budget(
                  *ciphertext, heongpu::ExecutionOptions().set_stream(stream)),
              expected_budget(1000));
    EXPECT_TRUE(started.load());
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}
