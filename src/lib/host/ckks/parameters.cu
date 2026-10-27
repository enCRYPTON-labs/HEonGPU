// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/parameters.cuh>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace heongpu
{
    namespace
    {
        constexpr int special_bit_size = MAX_USER_DEFINED_MOD_BIT_COUNT;

        int bit_length(Data64 value)
        {
            return 64 - __builtin_clzll(value);
        }

        int bit_count(const std::vector<Data64>& primes)
        {
            int count = 0;
            for (Data64 prime : primes)
            {
                count += bit_length(prime);
            }
            return count;
        }

        // Bit count of Q: the scaling primes alternate below and above
        // 2^scale_bit_size, starting below, and the ones above have one more
        // bit.
        int q_bit_count(const HEParams<Scheme::CKKS>& params, int depth)
        {
            return params.first_modulus_bit_size +
                   depth * params.scale_bit_size + depth / 2;
        }

        // The largest depth that fits limit with special_count P primes, or
        // -1 if none does.
        int max_depth(const HEParams<Scheme::CKKS>& params, int limit,
                      int special_count)
        {
            int depth = -1;
            while (q_bit_count(params, depth + 1) +
                       special_count * special_bit_size <=
                   limit)
            {
                depth++;
            }
            return depth;
        }

        void check(const HEParams<Scheme::CKKS>& params)
        {
            if (params.multiplicative_depth < 0)
            {
                throw ParameterError(error_message(
                    "Invalid multiplicative depth ",
                    params.multiplicative_depth, "! Expected 0 or more."));
            }
            if ((params.scale_bit_size < MIN_USER_DEFINED_MOD_BIT_COUNT) ||
                (params.scale_bit_size >= MAX_USER_DEFINED_MOD_BIT_COUNT))
            {
                throw ParameterError(error_message(
                    "Invalid scale bit size ", params.scale_bit_size,
                    "! Expected ", MIN_USER_DEFINED_MOD_BIT_COUNT, " to ",
                    MAX_USER_DEFINED_MOD_BIT_COUNT - 1, "."));
            }
            if ((params.first_modulus_bit_size < params.scale_bit_size) ||
                (params.first_modulus_bit_size >
                 MAX_USER_DEFINED_MOD_BIT_COUNT))
            {
                throw ParameterError(
                    error_message("Invalid first modulus bit size ",
                                  params.first_modulus_bit_size, "! Expected ",
                                  params.scale_bit_size, " to ",
                                  MAX_USER_DEFINED_MOD_BIT_COUNT, "."));
            }
            if (params.security_level == sec_level_type::none)
            {
                throw ParameterError(
                    "HEParams needs a security level! Set the ring size and "
                    "the moduli of the context directly for "
                    "sec_level_type::none.");
            }
            if ((params.poly_modulus_degree != 0) &&
                (max_coeff_bit_count(params.security_level,
                                     params.poly_modulus_degree) == 0))
            {
                throw ParameterError(error_message(
                    "Invalid poly modulus degree ", params.poly_modulus_degree,
                    "! Expected 0 or a power of two from ", MIN_POLY_DEGREE,
                    " to ", MAX_POLY_DEGREE, "."));
            }
            if (params.special_modulus_count < 0)
            {
                throw ParameterError(error_message(
                    "Invalid special modulus count ",
                    params.special_modulus_count, "! Expected 0 or more."));
            }
        }
    } // namespace

    int HEParameterSet<Scheme::CKKS>::multiplicative_depth() const
    {
        return static_cast<int>(coeff_modulus.size()) - 1;
    }

    int HEParameterSet<Scheme::CKKS>::total_bit_count() const
    {
        return bit_count(coeff_modulus) + bit_count(special_modulus);
    }

    int HEParameterSet<Scheme::CKKS>::max_bit_count() const
    {
        return max_coeff_bit_count(security_level, poly_modulus_degree);
    }

    int HEParameterSet<Scheme::CKKS>::digit_count() const
    {
        int q_count = static_cast<int>(coeff_modulus.size());
        int p_count = static_cast<int>(special_modulus.size());
        if (p_count == 0)
        {
            return 0;
        }
        return (q_count + p_count - 1) / p_count;
    }

    void HEParameterSet<Scheme::CKKS>::apply(
        const HEContext<Scheme::CKKS>& context) const
    {
        if (context->get_security_level() != security_level)
        {
            throw ParameterError(
                error_message("The context has the security level ",
                              detail::name(context->get_security_level()),
                              ", the parameters are for ",
                              detail::name(security_level), "!"));
        }
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_values(coeff_modulus, special_modulus);
    }

    HEContext<Scheme::CKKS> HEParameterSet<Scheme::CKKS>::make_context() const
    {
        auto context = GenHEContext<Scheme::CKKS>(security_level);
        apply(context);
        return context;
    }

    std::string HEParameterSet<Scheme::CKKS>::to_string() const
    {
        std::ostringstream stream;
        auto bits = [&](const std::vector<Data64>& primes)
        {
            for (std::size_t i = 0; i < primes.size(); i++)
            {
                stream << (i == 0 ? "" : "+") << bit_length(primes[i]);
            }
        };
        stream << "HEParameterSet<CKKS>(N=" << poly_modulus_degree << ", Q=";
        bits(coeff_modulus);
        stream << " bits, P=";
        bits(special_modulus);
        stream << " bits, total=" << total_bit_count() << "/" << max_bit_count()
               << " bits, digits=" << digit_count() << ", scale=2^"
               << std::log2(scale)
               << ", security=" << detail::name(security_level) << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEParameterSet<Scheme::CKKS>& parameters)
    {
        return os << parameters.to_string();
    }

    HEParameterSet<Scheme::CKKS> HEParams<Scheme::CKKS>::generate() const
    {
        check(*this);

        int depth = multiplicative_depth;
        int q_bits = q_bit_count(*this, depth);
        int min_special_count = std::max(special_modulus_count, 1);

        // The smallest ring size that fits Q and the P primes, or the given
        // one.
        std::size_t degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MIN_POLY_DEGREE;
        std::size_t max_degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MAX_POLY_DEGREE;
        while ((degree < max_degree) &&
               (q_bits + min_special_count * special_bit_size >
                max_coeff_bit_count(security_level, degree)))
        {
            degree *= 2;
        }

        int limit = max_coeff_bit_count(security_level, degree);
        if (q_bits + min_special_count * special_bit_size > limit)
        {
            int fitting_depth = max_depth(*this, limit, min_special_count);
            std::string special =
                (special_modulus_count == 0)
                    ? std::string("one P prime")
                    : error_message(special_modulus_count, " P primes");
            std::string ring =
                (poly_modulus_degree != 0)
                    ? error_message("N = ", degree)
                    : error_message("any N up to ", MAX_POLY_DEGREE);
            std::string fitting =
                (fitting_depth < 0)
                    ? std::string("No depth fits.")
                    : error_message("At most depth ", fitting_depth, " fits.");
            throw ParameterError(error_message(
                "Depth ", depth, " with the scale 2^", scale_bit_size,
                " does not fit ", detail::name(security_level), " with ", ring,
                ": Q and ", special, " need ",
                q_bits + min_special_count * special_bit_size,
                " bits, the limit is ", limit, ". ", fitting));
        }

        int special_count = special_modulus_count;
        if (special_count == 0)
        {
            special_count =
                std::min((limit - q_bits) / special_bit_size, depth + 1);
        }

        HEParameterSet<Scheme::CKKS> parameters;
        parameters.security_level = security_level;
        parameters.poly_modulus_degree = degree;
        parameters.scale = scale();

        parameters.coeff_modulus =
            generate_primes_below(degree, first_modulus_bit_size, 1);
        std::vector<Data64> scaling = generate_primes_around(
            degree, scale_bit_size, depth, parameters.coeff_modulus);
        parameters.coeff_modulus.insert(parameters.coeff_modulus.end(),
                                        scaling.begin(), scaling.end());
        parameters.special_modulus = generate_primes_below(
            degree, special_bit_size, special_count, parameters.coeff_modulus);

        return parameters;
    }

    HEContext<Scheme::CKKS> HEParams<Scheme::CKKS>::make_context() const
    {
        return generate().make_context();
    }

    double HEParams<Scheme::CKKS>::scale() const
    {
        return std::ldexp(1.0, scale_bit_size);
    }

} // namespace heongpu
