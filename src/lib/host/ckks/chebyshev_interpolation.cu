// Copyright 2025-2026 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

#include <cmath>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>

namespace heongpu
{
    // Generate Chebyshev interpolation nodes in interval [a, b]
    __host__ std::vector<double> chebyshev_nodes(int n, double a, double b)
    {
        std::vector<double> nodes(n);
        double x = 0.5 * (a + b);
        double y = 0.5 * (b - a);

        for (int k = 1; k <= n; k++)
        {
            nodes[k - 1] = x + y * std::cos((k - 0.5) * M_PI / n);
        }

        return nodes;
    }

    // Compute Chebyshev coefficients from function values at nodes
    __host__ std::vector<Complex64>
    chebyshev_coeffs(const std::vector<double>& nodes,
                     const std::vector<Complex64>& fi, double a, double b)
    {
        int n = nodes.size();
        std::vector<Complex64> coeffs(n, Complex64(0.0, 0.0));

        for (int i = 0; i < n; i++)
        {
            // Normalize node to [-1, 1]
            double u = (2.0 * nodes[i] - a - b) / (b - a);
            Complex64 Tprev(1.0, 0.0);
            Complex64 T(u, 0.0);
            Complex64 Tnext;

            for (int j = 0; j < n; j++)
            {
                coeffs[j] = coeffs[j] + fi[i] * Tprev;
                Tnext = Complex64(2.0 * u, 0.0) * T - Tprev;
                Tprev = T;
                T = Tnext;
            }
        }

        // Normalize coefficients
        coeffs[0] = coeffs[0] / Complex64(static_cast<double>(n), 0.0);
        for (int i = 1; i < n; i++)
        {
            coeffs[i] =
                coeffs[i] * Complex64(2.0 / static_cast<double>(n), 0.0);
        }

        return coeffs;
    }

    // Approximate a complex-valued function using Chebyshev interpolation
    __host__ std::vector<Complex64>
    approximate_function(std::function<Complex64(Complex64)> function, double a,
                         double b, int degree)
    {
        // Generate Chebyshev nodes
        std::vector<double> nodes = chebyshev_nodes(degree + 1, a, b);

        // Evaluate complex function at each node (nodes are real, but function
        // returns complex)
        std::vector<Complex64> fi(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++)
        {
            // Create complex number with real part = nodes[i], imag part = 0
            Complex64 input(nodes[i], 0.0);
            fi[i] = function(input);
        }

        // Compute and return Chebyshev coefficients
        return chebyshev_coeffs(nodes, fi, a, b);
    }

    __host__ std::vector<double>
    chebyshev_coefficients(const std::function<double(double)>& function,
                           double a, double b, int degree)
    {
        std::vector<double> nodes = chebyshev_nodes(degree + 1, a, b);
        std::vector<Complex64> fi(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++)
        {
            fi[i] = Complex64(function(nodes[i]), 0.0);
        }

        std::vector<Complex64> coeffs = chebyshev_coeffs(nodes, fi, a, b);
        std::vector<double> result(coeffs.size());
        for (size_t i = 0; i < coeffs.size(); i++)
        {
            result[i] = coeffs[i].real();
        }
        return result;
    }

} // namespace heongpu
