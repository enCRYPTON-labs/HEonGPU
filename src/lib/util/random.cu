// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/random.cuh>

namespace heongpu
{
    std::shared_ptr<rngongpu::RNG<rngongpu::Mode::AES>>
        RandomNumberGenerator::generator_ = nullptr;
    bool RandomNumberGenerator::initialized_ = false;
    std::mutex RandomNumberGenerator::mutex_;

    RandomNumberGenerator& RandomNumberGenerator::instance()
    {
        static RandomNumberGenerator instance;
        return instance;
    }

    void RandomNumberGenerator::initialize(
        const std::vector<unsigned char>& key,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        rngongpu::SecurityLevel security_level,
        bool prediction_resistance_enabled)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!initialized_)
        {
            generator_ = std::make_shared<rngongpu::RNG<rngongpu::Mode::AES>>(
                key, nonce, personalization_string, security_level,
                prediction_resistance_enabled);

            initialized_ = true;
        }
    }

    void RandomNumberGenerator::set(
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        cudaStream_t stream)
    {
        generator_->set(entropy_input, nonce, personalization_string, stream);
    }

    RandomNumberGenerator::RandomNumberGenerator() = default;

    RandomNumberGenerator::~RandomNumberGenerator()
    {
        generator_.reset();
    }

    __host__ void
    RandomNumberGenerator::modular_uniform_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_uniform_random_number(pointer, modulus, log_size,
                                                  mod_count, repeat_count,
                                                  additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_uniform_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_uniform_random_number(
            pointer, modulus, log_size, mod_count, repeat_count, entropy_input,
            additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_uniform_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_uniform_random_number(
            pointer, modulus, log_size, mod_count, mod_index, repeat_count,
            additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_uniform_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_uniform_random_number(
            pointer, modulus, log_size, mod_count, mod_index, repeat_count,
            entropy_input, additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_gaussian_random_number_generation(
        Float64 std_dev, Data64* pointer, Modulus64* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_normal_random_number(
            std_dev, pointer, modulus, log_size, mod_count, repeat_count,
            additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_gaussian_random_number_generation(
        Float64 std_dev, Data64* pointer, Modulus64* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_normal_random_number(
            std_dev, pointer, modulus, log_size, mod_count, repeat_count,
            entropy_input, additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_gaussian_random_number_generation(
        Float64 std_dev, Data64* pointer, Modulus64* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_normal_random_number(
            std_dev, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_gaussian_random_number_generation(
        Float64 std_dev, Data64* pointer, Modulus64* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_normal_random_number(
            std_dev, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, entropy_input, additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_ternary_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_ternary_random_number(pointer, modulus, log_size,
                                                  mod_count, repeat_count,
                                                  additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_ternary_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_ternary_random_number(
            pointer, modulus, log_size, mod_count, repeat_count, entropy_input,
            additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_ternary_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input = {};
        generator_->modular_ternary_random_number(
            pointer, modulus, log_size, mod_count, mod_index, repeat_count,
            additional_input, stream);
    }

    __host__ void
    RandomNumberGenerator::modular_ternary_random_number_generation(
        Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        generator_->modular_ternary_random_number(
            pointer, modulus, log_size, mod_count, mod_index, repeat_count,
            entropy_input, additional_input, stream);
    }

} // namespace heongpu
