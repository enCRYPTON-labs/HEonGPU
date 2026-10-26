// Copyright 2025-2026 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

#ifndef HEONGPU_CHEBYSHEV_INTERPOLATION_H
#define HEONGPU_CHEBYSHEV_INTERPOLATION_H

#include <vector>
#include <functional>
#include "gpufft/complex.cuh"

namespace heongpu
{
    /**
     * @brief Generate Chebyshev interpolation nodes in interval [a, b]
     *
     * @param n Number of nodes
     * @param a Lower bound of interval
     * @param b Upper bound of interval
     * @return std::vector<double> Vector of Chebyshev nodes
     */
    __host__ std::vector<double> chebyshev_nodes(int n, double a, double b);

    /**
     * @brief Compute Chebyshev coefficients from function values at nodes
     *
     * @param nodes Chebyshev interpolation nodes
     * @param fi Function values at nodes
     * @param a Lower bound of interval
     * @param b Upper bound of interval
     * @return std::vector<Complex64> Chebyshev coefficients
     */
    __host__ std::vector<Complex64>
    chebyshev_coeffs(const std::vector<double>& nodes,
                     const std::vector<Complex64>& fi, double a, double b);

    /**
     * @brief Approximate a complex-valued function using Chebyshev
     * interpolation
     *
     * @param function Function to approximate (Complex64 -> Complex64)
     * @param a Lower bound of interval
     * @param b Upper bound of interval
     * @param degree Degree of polynomial approximation
     * @return std::vector<Complex64> Chebyshev coefficients
     */
    __host__ std::vector<Complex64>
    approximate_function(std::function<Complex64(Complex64)> function, double a,
                         double b, int degree);

    /**
     * @brief Returns the coefficients of the Chebyshev interpolant of the
     * given degree of function on [a, b], for evaluate_chebyshev: the
     * polynomial that equals function at the degree + 1 Chebyshev nodes of
     * [a, b].
     */
    __host__ std::vector<double>
    chebyshev_coefficients(const std::function<double(double)>& function,
                           double a, double b, int degree);

} // namespace heongpu

#endif // HEONGPU_CHEBYSHEV_INTERPOLATION_H
