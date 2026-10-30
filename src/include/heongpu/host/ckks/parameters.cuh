// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_PARAMETERS_H
#define HEONGPU_CKKS_PARAMETERS_H

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <vector>
#include <heongpu/util/util.cuh>
#include <heongpu/host/ckks/context.cuh>

namespace heongpu
{
    /**
     * @brief CKKS parameters chosen by HEParams<Scheme::CKKS>::generate():
     * the ring size, the Q and P primes and the scale.
     */
    template <> struct HEParameterSet<Scheme::CKKS>
    {
        sec_level_type security_level = sec_level_type::sec128;
        std::size_t poly_modulus_degree = 0;
        // Q: the first prime, then one scaling prime per level. Rescaling
        // drops the last one first.
        std::vector<Data64> coeff_modulus;
        // P, used by key switching.
        std::vector<Data64> special_modulus;
        double scale = 0.0;
        // Set on the context by apply(), with scale as the scale of the top
        // level.
        heongpu::rescale_mode rescale_mode = heongpu::rescale_mode::manual;

        // The number of rescalings Q allows.
        int multiplicative_depth() const;

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
         * @brief Sets the ring size, the moduli and the rescale mode of
         * context, before its generate(). Throws ParameterError if the
         * security level of context is not security_level.
         */
        void apply(const HEContext<Scheme::CKKS>& context) const;

        /**
         * @brief Returns a new context with these parameters, not generated
         * yet.
         */
        HEContext<Scheme::CKKS> make_context() const;

        /**
         * @brief Returns a one line description: the ring size, the bit sizes
         * of the primes, the total bit count against the limit and the number
         * of key switching digits.
         */
        std::string to_string() const;
    };

    /**
     * @brief Chooses CKKS parameters from what the computation needs: the
     * multiplicative depth, the scale and the security level.
     *
     * generate() takes the smallest ring size N whose security limit fits Q,
     * which has a first prime of first_modulus_bit_size bits and one scaling
     * prime per level, together with at least one 60 bit P prime. Rescaling
     * by the scaling prime q_l turns the scale s_l of level l into
     * s_l^2 / q_l when ciphertexts of the same scale are multiplied, so q_l
     * is the prime closest to s_l^2 / 2^scale_bit_size: the scale stays
     * around 2^scale_bit_size at every level.
     * P then gets as many 60 bit primes as the limit leaves room for, up to
     * the number of Q primes: a larger P splits Q into fewer key switching
     * digits, which makes key switching faster and the keys smaller. Of
     * these it takes the fewest that give the same number of digits.
     *
     * @code
     * heongpu::HEParams<heongpu::Scheme::CKKS> params;
     * params.multiplicative_depth = 10;
     * auto context = params.make_context();
     * context->generate();
     * double scale = params.scale();
     * @endcode
     */
    template <> struct HEParams<Scheme::CKKS>
    {
        // Number of rescalings the computation needs.
        int multiplicative_depth = 1;
        // The scale is 2^scale_bit_size; from 30 to 59.
        int scale_bit_size = 40;
        // Bit size of the first Q prime, which bounds the result at level 0;
        // from scale_bit_size to 60.
        int first_modulus_bit_size = 60;
        // sec128, sec192 or sec256; none for no security limit, which needs
        // poly_modulus_degree and takes one P prime per Q prime unless
        // special_modulus_count is given.
        sec_level_type security_level = sec_level_type::sec128;
        // 0 to take the smallest ring size the security level allows.
        std::size_t poly_modulus_degree = 0;
        // Number of 60 bit P primes; 0 to take as many as fit.
        int special_modulus_count = 0;
        // The rescale mode of the context, see
        // HEContextImpl<Scheme::CKKS>::set_rescale_mode. Unset, generate()
        // takes rescale_mode::manual and HESession rescale_mode::automatic.
        std::optional<heongpu::rescale_mode> rescale_mode;
        // Moduli given instead of the ones generate() chooses: the bit sizes
        // of the Q and P primes, whose primes are generated as
        // HEContextImpl::set_coeff_modulus_bit_sizes does, or the primes
        // themselves, which need poly_modulus_degree. Either bit sizes or
        // primes, with at least one Q and one P modulus; multiplicative_depth,
        // first_modulus_bit_size and special_modulus_count keep their
        // defaults then. generate() takes the smallest ring size whose
        // security limit fits them, or poly_modulus_degree.
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
        HEParameterSet<Scheme::CKKS> generate() const;

        /**
         * @brief Returns a new context with the parameters of generate(), not
         * generated yet.
         */
        HEContext<Scheme::CKKS> make_context() const;

        // The scale to encode with, 2^scale_bit_size.
        double scale() const;
    };

    std::ostream& operator<<(std::ostream& os,
                             const HEParameterSet<Scheme::CKKS>& parameters);

} // namespace heongpu
#endif // HEONGPU_CKKS_PARAMETERS_H
