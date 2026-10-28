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

        // Q for the ring size degree: the first prime, then one scaling
        // prime per level. Rescaling by q_l turns the scale s_l of level l
        // into s_l^2 / q_l, the scale of level l - 1 when ciphertexts of the
        // same scale are multiplied. Each q_l is the prime closest to
        // s_l^2 / 2^scale_bit_size, so the scale stays around
        // 2^scale_bit_size at every level instead of doubling the error of
        // the level above (eprint 2020/1118). The levels are chosen from the
        // top, which rescaling drops first.
        std::vector<Data64> q_primes(const HEParams<Scheme::CKKS>& params,
                                     std::size_t degree, int depth)
        {
            std::vector<Data64> primes =
                generate_primes_below(degree, params.first_modulus_bit_size, 1);
            // The scale stays close to 2^scale_bit_size, so one more
            // candidate on each side than there are levels suffices.
            std::vector<Data64> candidates = generate_primes_around(
                degree, params.scale_bit_size, 2 * depth + 2, primes);

            std::vector<Data64> scaling(depth);
            double power = params.scale();
            double scale = power;
            for (int level = depth; level >= 1; level--)
            {
                double target = scale * scale / power;
                auto closest = std::min_element(
                    candidates.begin(), candidates.end(),
                    [&](Data64 a, Data64 b)
                    {
                        return std::abs(static_cast<double>(a) - target) <
                               std::abs(static_cast<double>(b) - target);
                    });
                scaling[level - 1] = *closest;
                scale = scale * scale / static_cast<double>(*closest);
                candidates.erase(closest);
            }
            primes.insert(primes.end(), scaling.begin(), scaling.end());
            return primes;
        }

        // The largest depth whose Q fits limit with special_count P primes
        // for the ring size degree, or -1 if none does.
        int max_depth(const HEParams<Scheme::CKKS>& params, std::size_t degree,
                      int limit, int special_count)
        {
            auto fits = [&](int depth)
            {
                return bit_count(q_primes(params, degree, depth)) +
                           special_count * special_bit_size <=
                       limit;
            };
            // Every scaling prime has at least scale_bit_size bits.
            int low = -1;
            int high = (limit - special_count * special_bit_size -
                        params.first_modulus_bit_size) /
                           params.scale_bit_size +
                       1;
            while (high - low > 1)
            {
                int middle = (low + high) / 2;
                if ((middle >= 0) && fits(middle))
                {
                    low = middle;
                }
                else
                {
                    high = middle;
                }
            }
            return low;
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
        context->set_rescale_mode(rescale_mode, scale);
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
               << ", security=" << detail::name(security_level);
        if (rescale_mode != heongpu::rescale_mode::manual)
        {
            stream << ", rescale=" << detail::name(rescale_mode);
        }
        stream << ")";
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
        int min_special_count = std::max(special_modulus_count, 1);

        // The smallest ring size that fits Q and the P primes, or the given
        // one. The primes depend on the ring size.
        std::size_t degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MIN_POLY_DEGREE;
        std::size_t max_degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MAX_POLY_DEGREE;
        // Every scaling prime has at least scale_bit_size bits, so smaller
        // ring sizes need not be tried.
        int min_bits = first_modulus_bit_size + depth * scale_bit_size +
                       min_special_count * special_bit_size;
        while ((degree < max_degree) &&
               (min_bits > max_coeff_bit_count(security_level, degree)))
        {
            degree *= 2;
        }
        std::vector<Data64> q;
        int limit;
        while (true)
        {
            q = q_primes(*this, degree, depth);
            limit = max_coeff_bit_count(security_level, degree);
            if ((bit_count(q) + min_special_count * special_bit_size <=
                 limit) ||
                (degree >= max_degree))
            {
                break;
            }
            degree *= 2;
        }

        int q_bits = bit_count(q);
        if (q_bits + min_special_count * special_bit_size > limit)
        {
            int fitting_depth =
                max_depth(*this, degree, limit, min_special_count);
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
            // As many P primes as fit, at most one per Q prime, then the
            // fewest that split Q into the same number of digits: more would
            // make the keys larger without making key switching faster.
            int q_count = depth + 1;
            int fitting =
                std::min((limit - q_bits) / special_bit_size, q_count);
            int digits = (q_count + fitting - 1) / fitting;
            special_count = (q_count + digits - 1) / digits;
        }

        HEParameterSet<Scheme::CKKS> parameters;
        parameters.security_level = security_level;
        parameters.poly_modulus_degree = degree;
        parameters.scale = scale();
        parameters.rescale_mode =
            rescale_mode.value_or(heongpu::rescale_mode::manual);

        parameters.coeff_modulus = q;
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
