// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_UTIL_H
#define HEONGPU_UTIL_H

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <heongpu/kernel/defines.h>
#include <unordered_map>
#include <stdexcept>
#include <set>
#include <heongpu/util/error.h>
#include <heongpu/util/serialization_header.h>
#include <heongpu/util/storagemanager.cuh>
#include <gmp.h>

namespace heongpu
{
    class CudaException : public std::exception
    {
      public:
        CudaException(const std::string& file, int line, cudaError_t error)
            : file_(file), line_(line), error_(error)
        {
        }

        const char* what() const noexcept override
        {
            return m_error_string.c_str();
        }

      private:
        std::string file_;
        int line_;
        cudaError_t error_;
        std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                     std::to_string(line_) + ": " +
                                     cudaGetErrorString(error_);
    };

#define HEONGPU_CUDA_CHECK(err)                                                \
    do                                                                         \
    {                                                                          \
        cudaError_t error = err;                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            throw CudaException(__FILE__, __LINE__, error);                    \
        }                                                                      \
    } while (0)

    bool coefficient_validator(const std::vector<int>& log_Q_bases_bit_sizes,
                               const std::vector<int>& log_P_bases_bit_sizes);

    // Throws ParameterError unless P has at least as many bits as every key
    // switching digit of Q: the Q primes in groups of as many primes as P
    // has, the last group possibly smaller.
    void validate_key_switching_digits(
        const std::vector<int>& log_Q_bases_bit_sizes,
        const std::vector<int>& log_P_bases_bit_sizes);

    /**
     * @brief Returns the largest total bit size of the Q and P primes that
     * level allows for the ring size poly_modulus_degree, the limit the
     * contexts check; 0 for sec_level_type::none and unsupported ring sizes.
     */
    int max_coeff_bit_count(sec_level_type level, size_t poly_modulus_degree);

    // Makes sure the per-thread kernel stack size is at least `bytes`, keeping
    // a larger limit set by the application. A smaller limit is raised when
    // `raise` is true and reported with StateError otherwise.
    void ensure_kernel_stack_size(size_t bytes, bool raise = true);

    namespace detail
    {
        // Names used by the to_string() functions.
        const char* name(sec_level_type level);
        const char* name(storage_type type);
        const char* name(encoding type);
        const char* name(rescale_mode mode);
    } // namespace detail

    // Identity of an HEContext, computed from the parameters its objects
    // depend on: the scheme, the ring size, the Q and P primes and the
    // plaintext modulus (0 when the scheme has none). Contexts generated from
    // the same parameters get the same identity. Never returns 0, which marks
    // objects whose context is unknown.
    std::uint64_t context_identity(int scheme, int ring_size,
                                   const std::vector<Modulus64>& primes,
                                   Data64 plain_modulus);

    // Throws ContextMismatchError when an object of the context with identity
    // `object_id` is used with `context`. Objects with identity 0 (for example
    // ones loaded from a stream) are not checked.
    template <typename Context>
    void validate_context_identity(std::uint64_t object_id,
                                   const Context& context,
                                   const char* object_name)
    {
        if (object_id == 0 || !context || object_id == context->context_id())
        {
            return;
        }
        std::ostringstream stream;
        stream << object_name << " context mismatch! The " << object_name
               << " belongs to the context with id 0x" << std::hex << object_id
               << " but is used with the context with id 0x"
               << context->context_id()
               << ". Objects can only be used with the context they were "
                  "created with or with a context generated from the same "
                  "parameters.";
        throw ContextMismatchError(stream.str());
    }

    // Throws ContextMismatchError when a key with `key_digits` key switching
    // digits is used with a context that splits Q into `context_digits`. The
    // context identity does not tell them apart: BFV keys of parameters with
    // more than two P primes, saved by HEonGPU 1.7.0 or earlier, have digits
    // of two primes.
    inline void validate_key_digit_count(int key_digits, int context_digits,
                                         const char* object_name)
    {
        if (key_digits == context_digits)
        {
            return;
        }
        std::ostringstream stream;
        stream << "The " << object_name << " has " << key_digits
               << " key switching digits, but the context splits Q into "
               << context_digits
               << "! It was generated for another split of Q; generate it "
                  "again with this context.";
        throw ContextMismatchError(stream.str());
    }

    enum class SineType
    {
        COS1,
    };

    enum class PolyType
    {
        MONOMIAL,
        CHEBYSHEV
    };

    enum class LinearTransformType
    {
        COEFFS_TO_SLOTS,
        SLOTS_TO_COEFFS
    };

    struct EvalModConfig
    {
        Data64 Q_;
        int level_start_;
        double message_ratio_;
        int K_;
        int sine_deg_;
        int double_angle_;
        int arcsine_deg_;
        double scaling_factor_;
        int piece_;
        double sqrt2pi_;
        double q_diff_;

        SineType sine_type_ = SineType::COS1;
        PolyType poly_type_ = PolyType::CHEBYSHEV;

        EvalModConfig()
            : sine_deg_(0), double_angle_(0), arcsine_deg_(0), level_start_(0),
              K_(0), message_ratio_(0.0), Q_(0), scaling_factor_(0.0), piece_(0)
        {
        }

        EvalModConfig(int level_start)
            : sine_deg_(30), double_angle_(3), arcsine_deg_(0),
              level_start_(level_start), K_(16), message_ratio_(256.0), Q_(0),
              scaling_factor_(0.0)
        {
        }

        EvalModConfig(Data64 Q, int level_start, double message_ratio, int K,
                      int sine_deg, int double_angle, int arcsine_deg,
                      double scaling_factor)
            : sine_deg_(sine_deg), double_angle_(double_angle),
              arcsine_deg_(arcsine_deg), level_start_(level_start), K_(K),
              message_ratio_(message_ratio), Q_(Q),
              scaling_factor_(scaling_factor)
        {
            // Calculate q_diff when Q is provided
            if (Q != 0)
            {
                q_diff_ =
                    static_cast<double>(Q) /
                    std::pow(2.0,
                             std::round(std::log2(static_cast<double>(Q))));
            }
            else
            {
                q_diff_ = 0.0;
            }

            // Calculate sqrt2pi when Q and scaling_factor are provided
            if (Q != 0 && scaling_factor != 0.0)
            {
                sqrt2pi_ = std::pow((static_cast<double>(Q) * 0.5) /
                                        (scaling_factor * M_PI),
                                    1.0 / std::pow(2.0, double_angle));
            }
            else
            {
                sqrt2pi_ = 0.0;
            }
        }
    };

    struct EncodingMatrixConfig
    {
        LinearTransformType lt_type_;
        int level_start_;
        double bsgs_ratio_;
        int piece_;

        EncodingMatrixConfig()
            : lt_type_(LinearTransformType::COEFFS_TO_SLOTS), level_start_(0),
              bsgs_ratio_(0.0), piece_(0)
        {
        }

        EncodingMatrixConfig(LinearTransformType lt_type, int level_start)
            : EncodingMatrixConfig(lt_type, level_start, 2.0)
        {
        }

        EncodingMatrixConfig(LinearTransformType lt_type, int level_start,
                             double bsgs_ratio)
            : lt_type_(lt_type), level_start_(level_start),
              bsgs_ratio_(bsgs_ratio)
        {
            piece_ = (lt_type == LinearTransformType::COEFFS_TO_SLOTS) ? 4 : 3;
        }

        EncodingMatrixConfig(LinearTransformType lt_type, int level_start,
                             double bsgs_ratio, int piece)
            : lt_type_(lt_type), level_start_(level_start),
              bsgs_ratio_(bsgs_ratio), piece_(piece)
        {
        }
    };

    struct BootstrappingConfig
    {
        int CtoS_piece_;
        int StoC_piece_;

        EncodingMatrixConfig cts_config_;
        EncodingMatrixConfig stc_config_;
        EvalModConfig eval_mod_config_;

        BootstrappingConfig(EncodingMatrixConfig stc_config,
                            EvalModConfig eval_mod_config,
                            EncodingMatrixConfig cts_config);
    };

    Data64 extendedGCD(Data64 a, Data64 b, Data64& x, Data64& y);
    Data64 modInverse(Data64 a, Data64 m);
    int calculate_bit_size(Data64 number);

    bool is_power_of_two(size_t number);
    int calculate_bit_count(Data64 number);
    int calculate_big_integer_bit_count(Data64* number, int word_count);

    /**
     * @brief Miller-Rabin test of the odd value with num_rounds bases. The
     * first twelve are the primes from 2 to 37, which decide every 64-bit
     * value without error; further rounds use random bases.
     */
    bool miller_rabin(const Data64& value, size_t num_rounds);

    bool is_prime(const Data64& value);

    std::vector<Data64> generate_proper_primes(Data64 factor, int bit_size,
                                               size_t count);

    std::vector<Modulus64>
    generate_primes(size_t poly_modulus_degree,
                    const std::vector<int> prime_bit_sizes);

    std::vector<Modulus64> generate_internal_primes(size_t poly_modulus_degree,
                                                    const int prime_count);

    /**
     * @brief Returns the count largest primes p = 1 (mod 2 *
     * poly_modulus_degree) below 2^bit_size that are not in excluded, largest
     * first. All of them have bit_size bits, from log2(poly_modulus_degree) + 2
     * to 61. Throws ParameterError if there are not enough of them.
     */
    std::vector<Data64>
    generate_primes_below(size_t poly_modulus_degree, int bit_size,
                          size_t count,
                          const std::vector<Data64>& excluded = {});

    /**
     * @brief Returns the count primes p = 1 (mod 2 * poly_modulus_degree)
     * closest to 2^bit_size that are not in excluded, alternately below and
     * above it, starting below. The ones below have bit_size bits, the ones
     * above bit_size + 1: the candidates for CKKS scaling primes. bit_size is
     * from log2(poly_modulus_degree) + 2 to 60. Throws ParameterError if there
     * are not enough of them.
     */
    std::vector<Data64>
    generate_primes_around(size_t poly_modulus_degree, int bit_size,
                           size_t count,
                           const std::vector<Data64>& excluded = {});

    bool is_primitive_root(Data64 root, size_t degree, Modulus64& modulus);

    bool find_primitive_root(size_t degree, Modulus64& modulus,
                             Data64& destination);

    Data64 find_minimal_primitive_root(size_t degree, Modulus64& modulus);

    std::vector<Data64>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus64> primes);

    std::vector<Root64>
    generate_ntt_table(std::vector<Data64> psi, std::vector<Modulus64> primes,
                       int n_power); // bit reverse order for GPU-NTT

    std::vector<Root64>
    generate_intt_table(std::vector<Data64> psi, std::vector<Modulus64> primes,
                        int n_power); // bit reverse order for GPU-NTT

    std::vector<Ninverse64> generate_n_inverse(size_t poly_modulus_degree,
                                               std::vector<Modulus64> primes);

    __global__ void unsigned_signed_convert(Data64* input, Data64* output,
                                            Modulus64* modulus);

    __global__ void fill_device_vector(Data64* vector, Data64 number, int size);

    std::vector<std::vector<int>> bsgs_index(const std::vector<int>& array,
                                             int N1, std::vector<int>& rot_n1,
                                             std::vector<int>& rot_n2);

    std::vector<std::vector<int>> seperate_func(const std::vector<int>& A,
                                                int slots,
                                                std::vector<int>& rot_n1,
                                                std::vector<int>& rot_n2,
                                                float bsgs_ratio);

    std::vector<int> unique_sort(const std::vector<int>& input);

    std::vector<Data64>
    calculate_last_q_modinv(const std::vector<Modulus64>& prime_vector,
                            const int Q_prime_size, const int P_size);

    std::vector<Data64>
    calculate_half(const std::vector<Modulus64>& prime_vector,
                   const int P_size);

    std::vector<Data64>
    calculate_half_mod(const std::vector<Modulus64>& prime_vector,
                       const std::vector<Data64>& half, const int Q_prime_size,
                       const int P_size);

    std::vector<Data64>
    calculate_factor(const std::vector<Modulus64>& prime_vector,
                     const int Q_size, const int P_size);

    std::vector<Data64> calculate_Mi(const std::vector<Modulus64>& prime_vector,
                                     const int size);

    std::vector<Data64>
    calculate_Mi_inv(const std::vector<Modulus64>& prime_vector,
                     const int size);

    std::vector<Data64> calculate_M(const std::vector<Modulus64>& prime_vector,
                                    const int size);

    std::vector<Data64>
    calculate_upper_half_threshold(const std::vector<Modulus64>& prime_vector,
                                   const int size);

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    static __device__ __forceinline__ uint32_t warp_reduce(uint32_t input)
    {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
#if defined(__CUDA_ARCH__)
            input += __shfl_down_sync(0xFFFFFFFF, input, offset);
#endif
        }
        return input;
    }

} // namespace heongpu
#endif // HEONGPU_UTIL_H
