// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/parameters.cuh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <sstream>

namespace heongpu
{
    namespace
    {
        constexpr int max_plain_modulus_bit_size = 30;
        // Bits of Q left for the noise after depth multiplications.
        constexpr int noise_margin = 10;

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

        // The noise model, calibrated against remainder_noise_budget for N
        // from 2^12 to 2^16, plaintext moduli of 17 to 30 bits and Q of 4 to
        // 14 primes: a fresh ciphertext uses up to log2(t) + ceil(log2(N) / 2)
        // + 2 bits of Q, and each multiplication with relinearization
        // log2(t) + log2(N) - 0.3 more on average, up to 1.4 more for a
        // single one. The model takes 0.5 more.
        int noise_bit_count(std::size_t degree, Data64 plain_modulus, int depth)
        {
            double log_t = std::log2(static_cast<double>(plain_modulus));
            int log_n = calculate_bit_size(degree) - 1;
            double fresh = log_t + (log_n + 1) / 2 + 2;
            double multiplication = log_t + log_n + 0.5;
            return static_cast<int>(std::ceil(fresh + depth * multiplication)) +
                   noise_margin;
        }

        // The bit sizes of Q: as few primes of up to 60 bits as possible,
        // whose bit sizes differ by at most one, largest first.
        std::vector<int> q_bit_sizes(int noise_bits)
        {
            noise_bits = std::max(noise_bits, MIN_USER_DEFINED_MOD_BIT_COUNT);
            int count = (noise_bits + MAX_USER_DEFINED_MOD_BIT_COUNT - 1) /
                        MAX_USER_DEFINED_MOD_BIT_COUNT;
            std::vector<int> sizes(count, noise_bits / count);
            for (int i = 0; i < noise_bits % count; i++)
            {
                sizes[i]++;
            }
            return sizes;
        }

        // Key switching splits Q into digits of as many primes as P has, so
        // P primes with a bit more than the largest Q prime suffice.
        int special_bit_size(const std::vector<int>& q_sizes)
        {
            return std::min(q_sizes[0] + 1, MAX_USER_DEFINED_MOD_BIT_COUNT);
        }

        // As many P primes as fit limit next to Q, at most one per Q prime,
        // then the fewest that split Q into the same number of digits: more
        // would make the keys larger without making key switching faster.
        int fitting_special_count(const std::vector<int>& q_sizes, int limit)
        {
            int q_count = static_cast<int>(q_sizes.size());
            int q_bits = 0;
            for (int size : q_sizes)
            {
                q_bits += size;
            }
            int count =
                std::min((limit - q_bits) / special_bit_size(q_sizes), q_count);
            int digits = (q_count + count - 1) / count;
            return (q_count + digits - 1) / digits;
        }

        // Bit count of Q and the P prime.
        int total_bit_count_for(std::size_t degree, Data64 plain_modulus,
                                int depth)
        {
            std::vector<int> sizes =
                q_bit_sizes(noise_bit_count(degree, plain_modulus, depth));
            int count = 0;
            for (int size : sizes)
            {
                count += size;
            }
            return count + special_bit_size(sizes);
        }

        // The largest prime 1 (mod 2N) of bit_size bits, or 0 if there is
        // none.
        Data64 batching_plain_modulus(std::size_t degree, int bit_size)
        {
            try
            {
                return generate_primes_below(degree, bit_size, 1)[0];
            }
            catch (const ParameterError&)
            {
                return 0;
            }
        }

        void check(const HEParams<Scheme::BFV>& params)
        {
            if (params.multiplicative_depth < 0)
            {
                throw ParameterError(error_message(
                    "Invalid multiplicative depth ",
                    params.multiplicative_depth, "! Expected 0 or more."));
            }
            if ((params.plain_modulus == 1) ||
                (params.plain_modulus > static_cast<Data64>(INT_MAX)))
            {
                throw ParameterError(error_message(
                    "Invalid plaintext modulus ", params.plain_modulus,
                    "! Expected 0 or 2 to ", INT_MAX, "."));
            }
            int min_plain_bit_size = calculate_bit_size(MIN_POLY_DEGREE) + 1;
            if ((params.plain_modulus == 0) &&
                ((params.plain_modulus_bit_size < min_plain_bit_size) ||
                 (params.plain_modulus_bit_size > max_plain_modulus_bit_size)))
            {
                throw ParameterError(
                    error_message("Invalid plaintext modulus bit size ",
                                  params.plain_modulus_bit_size, "! Expected ",
                                  min_plain_bit_size, " to ",
                                  max_plain_modulus_bit_size, "."));
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
        }
    } // namespace

    int HEParameterSet<Scheme::BFV>::total_bit_count() const
    {
        return bit_count(coeff_modulus) + bit_count(special_modulus);
    }

    int HEParameterSet<Scheme::BFV>::max_bit_count() const
    {
        return max_coeff_bit_count(security_level, poly_modulus_degree);
    }

    int HEParameterSet<Scheme::BFV>::digit_count() const
    {
        int q_count = static_cast<int>(coeff_modulus.size());
        int p_count = static_cast<int>(special_modulus.size());
        if (p_count == 0)
        {
            return 0;
        }
        return (q_count + p_count - 1) / p_count;
    }

    void HEParameterSet<Scheme::BFV>::apply(
        const HEContext<Scheme::BFV>& context) const
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
        context->set_plain_modulus(static_cast<int>(plain_modulus));
    }

    HEContext<Scheme::BFV> HEParameterSet<Scheme::BFV>::make_context() const
    {
        auto context = GenHEContext<Scheme::BFV>(security_level);
        apply(context);
        return context;
    }

    std::string HEParameterSet<Scheme::BFV>::to_string() const
    {
        std::ostringstream stream;
        auto bits = [&](const std::vector<Data64>& primes)
        {
            for (std::size_t i = 0; i < primes.size(); i++)
            {
                stream << (i == 0 ? "" : "+") << bit_length(primes[i]);
            }
        };
        stream << "HEParameterSet<BFV>(N=" << poly_modulus_degree << ", Q=";
        bits(coeff_modulus);
        stream << " bits, P=";
        bits(special_modulus);
        stream << " bits, total=" << total_bit_count() << "/" << max_bit_count()
               << " bits, digits=" << digit_count() << ", t=" << plain_modulus
               << ", security=" << detail::name(security_level) << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEParameterSet<Scheme::BFV>& parameters)
    {
        return os << parameters.to_string();
    }

    HEParameterSet<Scheme::BFV> HEParams<Scheme::BFV>::generate() const
    {
        check(*this);

        int depth = multiplicative_depth;

        // The smallest ring size that fits Q and the P prime, or the given
        // one. A plaintext modulus for batching depends on the ring size.
        std::size_t degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MIN_POLY_DEGREE;
        std::size_t max_degree =
            (poly_modulus_degree != 0) ? poly_modulus_degree : MAX_POLY_DEGREE;
        Data64 t;
        int limit;
        while (true)
        {
            t = (plain_modulus != 0)
                    ? plain_modulus
                    : batching_plain_modulus(degree, plain_modulus_bit_size);
            if (t == 0)
            {
                // Smaller ring sizes did not fit; larger ones need even
                // larger plaintext moduli.
                std::string smaller =
                    ((poly_modulus_degree == 0) && (degree > MIN_POLY_DEGREE))
                        ? error_message("Depth ", depth, " does not fit ",
                                        detail::name(security_level),
                                        " with N up to ", degree / 2,
                                        ", and no")
                        : std::string("No");
                throw ParameterError(error_message(
                    smaller, " prime 1 mod ", 2 * degree, " has ",
                    plain_modulus_bit_size, " bits! Batching with N = ", degree,
                    " needs a plaintext modulus of at least ",
                    calculate_bit_size(degree) + 1, " bits."));
            }
            limit = max_coeff_bit_count(security_level, degree);
            if ((total_bit_count_for(degree, t, depth) <= limit) ||
                (degree >= max_degree))
            {
                break;
            }
            degree *= 2;
        }

        int needed = total_bit_count_for(degree, t, depth);
        if (needed > limit)
        {
            int fitting_depth = -1;
            while (total_bit_count_for(degree, t, fitting_depth + 1) <= limit)
            {
                fitting_depth++;
            }
            std::string ring =
                (poly_modulus_degree != 0)
                    ? error_message("N = ", degree)
                    : error_message("any N up to ", MAX_POLY_DEGREE);
            std::string fitting =
                (fitting_depth < 0)
                    ? std::string("No depth fits.")
                    : error_message("At most depth ", fitting_depth, " fits.");
            throw ParameterError(error_message(
                "Depth ", depth, " with the plaintext modulus ", t,
                " does not fit ", detail::name(security_level), " with ", ring,
                ": Q and one P prime need ", needed, " bits, the limit is ",
                limit, ". ", fitting));
        }

        std::vector<int> sizes = q_bit_sizes(noise_bit_count(degree, t, depth));

        HEParameterSet<Scheme::BFV> parameters;
        parameters.security_level = security_level;
        parameters.poly_modulus_degree = degree;
        parameters.plain_modulus = t;
        std::vector<Data64> excluded = {t};
        for (std::size_t i = 0; i < sizes.size();)
        {
            // The primes of one bit size together.
            std::size_t end = i;
            while ((end < sizes.size()) && (sizes[end] == sizes[i]))
            {
                end++;
            }
            std::vector<Data64> primes =
                generate_primes_below(degree, sizes[i], end - i, excluded);
            parameters.coeff_modulus.insert(parameters.coeff_modulus.end(),
                                            primes.begin(), primes.end());
            excluded.insert(excluded.end(), primes.begin(), primes.end());
            i = end;
        }
        parameters.special_modulus = generate_primes_below(
            degree, special_bit_size(sizes),
            fitting_special_count(sizes, limit), excluded);

        return parameters;
    }

    HEContext<Scheme::BFV> HEParams<Scheme::BFV>::make_context() const
    {
        return generate().make_context();
    }

} // namespace heongpu
