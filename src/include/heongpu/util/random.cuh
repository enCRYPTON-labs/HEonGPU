// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_RANDOM_GENERATOR_CLASS_H
#define HEONGPU_RANDOM_GENERATOR_CLASS_H

#include <mutex>
#include <memory>
#include <vector>
#include <sys/sysinfo.h>
#include <heongpu/util/error.h>
#include "gpuntt/common/common.cuh"
#include "gpufft/complex.cuh"
#include "rngongpu/rand_aes/aes_rng.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

namespace heongpu
{
    struct RNGSeed
    {
        std::vector<unsigned char> key_;
        std::vector<unsigned char> nonce_;
        std::vector<unsigned char> personalization_string_;

        RNGSeed()
        {
            key_ = std::vector<unsigned char>(16); // for 128 bit
            if (1 != RAND_bytes(key_.data(), key_.size()))
                throw SystemError("RAND_bytes failed! OpenSSL could not "
                                  "generate random bytes.");
            nonce_ = std::vector<unsigned char>(8); // for 128 bit
            if (1 != RAND_bytes(nonce_.data(), nonce_.size()))
                throw SystemError("RAND_bytes failed! OpenSSL could not "
                                  "generate random bytes.");
        }

        RNGSeed(const std::vector<unsigned char>& key,
                const std::vector<unsigned char>& nonce,
                const std::vector<unsigned char>& personalization_string)
            : key_(key), nonce_(nonce),
              personalization_string_(personalization_string)
        {
            if (key_.size() < 16)
            {
                throw std::invalid_argument("Invalid key size!");
            }
        }
    };

    class RandomNumberGenerator
    {
      public:
        static RandomNumberGenerator& instance();

        void
        initialize(const std::vector<unsigned char>& key,
                   const std::vector<unsigned char>& nonce,
                   const std::vector<unsigned char>& personalization_string,
                   rngongpu::SecurityLevel security_level,
                   bool prediction_resistance_enabled);

        ~RandomNumberGenerator();

        void set(const std::vector<unsigned char>& entropy_input,
                 const std::vector<unsigned char>& nonce,
                 const std::vector<unsigned char>& personalization_string,
                 cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order. (From RNGonGPU Library)
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q0, array3 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_uniform_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int repeat_count, cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order. (From RNGonGPU Library)
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q0, array3 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_uniform_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order. (From RNGonGPU Library)
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0, array1]
         *
         *   - output array : [array0 % q0, array1 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q3, array2 % q0, array3 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_uniform_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order. (From RNGonGPU Library)
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0, array1]
         *
         *   - output array : [array0 % q0, array1 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q3, array2 % q0, array3 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_uniform_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order. (From RNGonGPU Library)
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_gaussian_random_number_generation(
            Float64 std_dev, Data64* pointer, Modulus64* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order. (From RNGonGPU Library)
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - array order  : [array0, array1] since repeat_count = 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_gaussian_random_number_generation(
            Float64 std_dev, Data64* pointer, Modulus64* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order. (From RNGonGPU Library)
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_gaussian_random_number_generation(
            Float64 std_dev, Data64* pointer, Modulus64* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order. (From RNGonGPU Library)
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_gaussian_random_number_generation(
            Float64 std_dev, Data64* pointer, Modulus64* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order.(-1,0,1) (From RNGonGPU Library)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_ternary_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int repeat_count, cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order.(-1,0,1) (From RNGonGPU Library)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_ternary_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order. (-1,0,1) (From RNGonGPU Library)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_ternary_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order. (-1,0,1) (From RNGonGPU Library)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        __host__ void modular_ternary_random_number_generation(
            Data64* pointer, Modulus64* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

      private:
        RandomNumberGenerator();
        RandomNumberGenerator(const RandomNumberGenerator&) = delete;
        RandomNumberGenerator& operator=(const RandomNumberGenerator&) = delete;

        static std::shared_ptr<rngongpu::RNG<rngongpu::Mode::AES>> generator_;
        static bool initialized_;
        static std::mutex mutex_;
    };

} // namespace heongpu
#endif // HEONGPU_RANDOM_GENERATOR_CLASS_H
