// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_PARAMETERS_H
#define HEONGPU_BFV_PARAMETERS_H

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>
#include <heongpu/util/util.cuh>
#include <heongpu/host/bfv/context.cuh>

namespace heongpu
{
    /**
     * @brief BFV parameters chosen by HEParams<Scheme::BFV>::generate(): the
     * ring size, the Q and P primes and the plaintext modulus.
     */
    template <> struct HEParameterSet<Scheme::BFV>
    {
        sec_level_type security_level = sec_level_type::sec128;
        std::size_t poly_modulus_degree = 0;
        // Q, primes whose bit sizes differ by at most one.
        std::vector<Data64> coeff_modulus;
        // P, used by key switching.
        std::vector<Data64> special_modulus;
        Data64 plain_modulus = 0;

        // The sum of the bit sizes of the Q and P primes, which the context
        // compares with max_bit_count().
        int total_bit_count() const;

        // The largest total bit count security_level allows for
        // poly_modulus_degree; 0 for sec_level_type::none, which has no
        // limit.
        int max_bit_count() const;

        // The number of digits key switching splits Q into: Q has
        // coeff_modulus.size() primes and each digit up to
        // special_modulus.size() of them.
        int digit_count() const;

        /**
         * @brief Sets the ring size, the moduli and the plaintext modulus of
         * context, before its generate(). Throws ParameterError if the
         * security level of context is not security_level.
         */
        void apply(const HEContext<Scheme::BFV>& context) const;

        /**
         * @brief Returns a new context with these parameters, not generated
         * yet.
         */
        HEContext<Scheme::BFV> make_context() const;

        /**
         * @brief Returns a one line description: the ring size, the bit sizes
         * of the primes, the total bit count against the limit, the number of
         * key switching digits and the plaintext modulus.
         */
        std::string to_string() const;
    };

    /**
     * @brief Chooses BFV parameters from what the computation needs: the
     * plaintext modulus, the multiplicative depth and the security level.
     *
     * Q gets the bits the noise of depth multiplications needs, by a model
     * calibrated against HEDecryptor::remainder_noise_budget, plus a margin of
     * 10 bits: a fresh ciphertext uses about log2(t) + log2(N) / 2 + 2 bits
     * of Q, and each multiplication with relinearization about
     * log2(t) + log2(N) more. generate() takes the smallest ring size N whose
     * security limit fits Q, in as few primes of up to 60 bits as possible,
     * and one P prime with a bit more than them. Then it takes as many of
     * these P primes as fit the limit, at most one per Q prime, and of those
     * the fewest that split Q into the same number of key switching digits:
     * a digit has as many Q primes as P has, and fewer digits make the keys
     * smaller and key switching faster.
     *
     * @code
     * heongpu::HEParams<heongpu::Scheme::BFV> params;
     * params.plain_modulus_bit_size = 20;
     * params.multiplicative_depth = 4;
     * auto context = params.make_context();
     * context->generate();
     * @endcode
     */
    template <> struct HEParams<Scheme::BFV>
    {
        // Number of multiplications in a row the computation needs. A
        // multiplication by a plaintext can use as much noise budget as one by
        // a ciphertext, less if the plaintext is small, for example one value
        // in every slot.
        int multiplicative_depth = 1;
        // The plaintext modulus t, from 2 to 2^31 - 1; 0 to take the largest
        // prime 1 (mod 2N) of plain_modulus_bit_size bits, which allows
        // batching. A given t allows batching if it is a prime 1 (mod 2N).
        Data64 plain_modulus = 0;
        // Bit size of the plaintext modulus when plain_modulus is 0; from
        // log2(N) + 2 to 30.
        int plain_modulus_bit_size = 20;
        // sec128, sec192 or sec256; none for no security limit, which needs
        // poly_modulus_degree and takes one P prime per Q prime unless
        // special_modulus_count is given.
        sec_level_type security_level = sec_level_type::sec128;
        // 0 to take the smallest ring size the security level allows.
        std::size_t poly_modulus_degree = 0;
        // Number of P primes, at most the number of Q primes unless
        // security_level is none; 0 to take as many as fit, see above.
        int special_modulus_count = 0;
        // Moduli given instead of the ones generate() chooses: the bit sizes
        // of the Q and P primes, whose primes are generated as
        // HEContextImpl::set_coeff_modulus_bit_sizes does, or the primes
        // themselves, which need poly_modulus_degree. Either bit sizes or
        // primes, with at least one Q and one P modulus; multiplicative_depth
        // and special_modulus_count keep their defaults then. generate() takes
        // the smallest ring size whose security limit fits them, or
        // poly_modulus_degree.
        std::vector<int> coeff_modulus_bit_sizes;
        std::vector<int> special_modulus_bit_sizes;
        std::vector<Data64> coeff_modulus;
        std::vector<Data64> special_modulus;

        /**
         * @brief Returns the parameters. Throws ParameterError if no
         * supported ring size fits them, or the given poly_modulus_degree or
         * special_modulus_count does not; the message gives the largest depth
         * that fits.
         */
        HEParameterSet<Scheme::BFV> generate() const;

        /**
         * @brief Returns a new context with the parameters of generate(), not
         * generated yet.
         */
        HEContext<Scheme::BFV> make_context() const;
    };

    std::ostream& operator<<(std::ostream& os,
                             const HEParameterSet<Scheme::BFV>& parameters);

} // namespace heongpu
#endif // HEONGPU_BFV_PARAMETERS_H
