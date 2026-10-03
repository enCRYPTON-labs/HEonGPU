// Copyright 2025-2026 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

// An implementation of ApproximateCos
// Original algorithm: Han and Ki, "Better Bootstrapping for Approximate
// Homomorphic Encryption" https://eprint.iacr.org/2019/688
//
// High-precision computation using NTL library (similar to
// github.com/DohyeongKi/better-homomorphic-sine-evaluation)

#include <NTL/RR.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <heongpu/host/ckks/cosine_approx.cuh>

using namespace NTL;

namespace heongpu
{

// High-precision PI constant
#define M_PIl 3.141592653589793238462643383279502884L

    // BigintCos: Iterative arbitrary precision computation of Cos(x)
    // Johansson, B. Tomas, "An elementary algorithm to evaluate trigonometric
    // functions to high precision", 2018 Iterative process with an error of
    // ~10^{-0.60206*k} after k iterations.
    RR BigintCos(const RR& x)
    {
        int k = 1000; // number of iterations
        RR t = RR(0.5);
        RR half = RR(0.5);

        // t = 0.5^(k-1)
        for (int i = 1; i < k - 1; i++)
        {
            t *= half;
        }

        // s = x * t * x * t
        RR s = x * t * x * t;

        RR four = RR(4.0);

        // Iterative computation
        for (int i = 1; i < k; i++)
        {
            s = s * (four - s);
        }

        // result = 1 - s/2
        RR result = RR(1.0) - s / RR(2.0);
        return result;
    }

    // Helper: Find index of maximum value in array
    int maxIndex(const std::vector<double>& array)
    {
        int maxind = 0;
        double max_val = array[0];
        for (size_t i = 1; i < array.size(); i++)
        {
            if (array[i] > max_val)
            {
                maxind = i;
                max_val = array[i];
            }
        }
        return maxind;
    }

    // genDegrees: Compute adaptive degree allocation for each integer interval
    // Returns: pair of (degree array, total degree)
    std::pair<std::vector<int>, int> genDegrees(int degree, int K, double dev)
    {
        int degbdd = degree + 1;
        int totdeg = 2 * K - 1;
        double err = 1.0 / dev;

        std::vector<int> deg(K, 1);
        std::vector<double> bdd(K);

        // Initialize error bound array
        double temp = 0.0;
        for (int i = 1; i <= 2 * K - 1; i++)
        {
            temp -= std::log2(static_cast<double>(i));
        }
        temp += (2.0 * K - 1.0) * std::log2(2.0 * M_PIl);
        temp += std::log2(err);

        for (int i = 0; i < K; i++)
        {
            bdd[i] = temp;
            for (int j = 1; j <= K - 1 - i; j++)
            {
                bdd[i] += std::log2(static_cast<double>(j) + err);
            }
            for (int j = 1; j <= K - 1 + i; j++)
            {
                bdd[i] += std::log2(static_cast<double>(j) + err);
            }
        }

        int maxiter = 200;

        for (int iter = 0; iter < maxiter; iter++)
        {
            if (totdeg >= degbdd)
            {
                break;
            }

            int maxi = maxIndex(bdd);

            if (maxi != 0)
            {
                if (totdeg + 2 > degbdd)
                {
                    break;
                }

                for (int i = 0; i < K; i++)
                {
                    bdd[i] -= std::log2(static_cast<double>(totdeg + 1));
                    bdd[i] -= std::log2(static_cast<double>(totdeg + 2));
                    bdd[i] += 2.0 * std::log2(2.0 * M_PIl);

                    if (i != maxi)
                    {
                        bdd[i] += std::log2(
                            std::abs(static_cast<double>(i - maxi)) + err);
                        bdd[i] +=
                            std::log2(static_cast<double>(i + maxi) + err);
                    }
                    else
                    {
                        bdd[i] += std::log2(err) - 1.0;
                        bdd[i] += std::log2(2.0 * static_cast<double>(i) + err);
                    }
                }

                totdeg += 2;
            }
            else
            {
                bdd[0] -= std::log2(static_cast<double>(totdeg + 1));
                bdd[0] += std::log2(err) - 1.0;
                bdd[0] += std::log2(2.0 * M_PIl);

                for (int i = 1; i < K; i++)
                {
                    bdd[i] -= std::log2(static_cast<double>(totdeg + 1));
                    bdd[i] += std::log2(2.0 * M_PIl);
                    bdd[i] += std::log2(static_cast<double>(i) + err);
                }

                totdeg++;
            }

            deg[maxi]++;
        }

        return {deg, totdeg};
    }

    // genNodes: Generate optimized interpolation nodes and compute Newton
    // divided differences Returns: pair of (Chebyshev evaluation points,
    // function values at those points, totdeg)
    std::tuple<std::vector<RR>, std::vector<RR>, int>
    genNodes(const std::vector<int>& deg, double dev, int totdeg, int K,
             int scnum, int period, double shift)
    {
        RR PI = RR(M_PIl);
        RR scfac = RR(period * (1 << scnum));
        RR intersize = RR(1.0 / dev);

        // Allocate interpolation nodes
        std::vector<RR> z(totdeg);
        int cnt = 0;

        // Add zero node if deg[0] is odd
        if (deg[0] % 2 != 0)
        {
            z[cnt] = RR(0.0);
            cnt++;
        }

        // Generate nodes for each integer interval
        for (int i = K - 1; i > 0; i--)
        {
            for (int j = 1; j <= deg[i]; j++)
            {
                // temp = cos((2*j-1)*PI / (2*deg[i]))
                RR temp = RR(2 * j - 1) * PI / RR(2 * deg[i]);
                temp = BigintCos(temp);
                temp = temp * intersize;

                // z[cnt] = i + temp
                z[cnt] = RR(i) + temp;
                cnt++;

                // z[cnt] = -i - temp
                z[cnt] = RR(-i) - temp;
                cnt++;
            }
        }

        // Nodes around zero
        for (int j = 1; j <= deg[0] / 2; j++)
        {
            RR temp = RR(2 * j - 1) * PI / RR(2 * deg[0]);
            temp = BigintCos(temp);
            temp = temp * intersize;

            z[cnt] = temp;
            cnt++;
            z[cnt] = -temp;
            cnt++;
        }

        // Compute d = cos(2*pi*(z-shift)/scfac) for Newton interpolation
        std::vector<RR> d(totdeg);
        for (int i = 0; i < totdeg; i++)
        {
            // Transform node: z[i] = (z[i] - shift) / scfac
            z[i] = (z[i] - RR(shift)) / scfac;

            // d[i] = cos(2 * PI * z[i])
            d[i] = BigintCos(RR(2.0) * PI * z[i]);
        }

        // Compute Newton divided differences
        for (int j = 1; j < totdeg; j++)
        {
            for (int l = 0; l < totdeg - j; l++)
            {
                d[l] = (d[l + 1] - d[l]) / (z[l + j] - z[l]);
            }
        }

        totdeg++;

        // Generate Chebyshev evaluation points x
        std::vector<RR> x(totdeg);
        for (int i = 0; i < totdeg; i++)
        {
            // x[i] = K/scfac * cos(i*PI/(totdeg-1))
            RR temp = RR(i) * PI / RR(totdeg - 1);
            x[i] = (RR(K) / scfac) * BigintCos(temp);
        }

        // Evaluate Newton polynomial at Chebyshev points
        std::vector<RR> p(totdeg);
        for (int i = 0; i < totdeg; i++)
        {
            p[i] = d[0];
            for (int j = 1; j < totdeg - 1; j++)
            {
                p[i] = p[i] * (x[i] - z[j]) + d[j];
            }
        }

        return {x, p, totdeg};
    }

    // ApproximateCos: Main function
    std::vector<std::complex<double>> ApproximateCos(int K, int degree,
                                                     double dev, int scnum,
                                                     int period, double shift)
    {
        // Set NTL precision (1000 bits, same as Go's big.Float precision)
        RR::SetPrecision(1000);

        RR PI = RR(M_PIl);
        RR scfac = RR(period * (1 << scnum));

        // Step 1: Generate adaptive degrees
        auto [deg, totdeg] = genDegrees(degree, K, dev);

        // Step 2: Generate nodes and Newton coefficients
        auto [x, p, final_totdeg] =
            genNodes(deg, dev, totdeg, K, scnum, period, shift);
        totdeg = final_totdeg;

        // Step 3: Build Chebyshev basis matrix T
        // T[i][j] = T_j(x[i] / (K/scfac)) where T_j is the j-th Chebyshev
        // polynomial
        std::vector<std::vector<RR>> T(totdeg, std::vector<RR>(totdeg));

        for (int i = 0; i < totdeg; i++)
        {
            // T[i][0] = 1
            T[i][0] = RR(1.0);

            // T[i][1] = x[i] / (K/scfac)
            RR normalized_x = x[i] / (RR(K) / scfac);
            T[i][1] = normalized_x;

            // T[i][j] = 2 * normalized_x * T[i][j-1] - T[i][j-2]
            for (int j = 2; j < totdeg; j++)
            {
                T[i][j] = RR(2.0) * normalized_x * T[i][j - 1] - T[i][j - 2];
            }
        }

        // Step 4: Gaussian elimination with partial pivoting (same as GitHub
        // implementation)
        for (int i = 0; i < totdeg - 1; i++)
        {
            // Find pivot
            RR maxabs = abs(T[i][i]);
            int maxindex = i;

            for (int j = i + 1; j < totdeg; j++)
            {
                if (abs(T[j][i]) > maxabs)
                {
                    maxabs = abs(T[j][i]);
                    maxindex = j;
                }
            }

            // Swap rows if needed
            if (i != maxindex)
            {
                std::swap(T[i], T[maxindex]);
                std::swap(p[i], p[maxindex]);
            }

            // Scale row i
            for (int j = i + 1; j < totdeg; j++)
            {
                T[i][j] /= T[i][i];
            }
            p[i] /= T[i][i];
            T[i][i] = RR(1.0);

            // Eliminate column i
            for (int j = i + 1; j < totdeg; j++)
            {
                p[j] -= T[j][i] * p[i];
                for (int l = i + 1; l < totdeg; l++)
                {
                    T[j][l] -= T[j][i] * T[i][l];
                }
                T[j][i] = RR(0.0);
            }
        }

        // Step 5: Back substitution to get Chebyshev coefficients
        std::vector<RR> c(totdeg);
        c[totdeg - 1] = p[totdeg - 1];

        for (int i = totdeg - 2; i >= 0; i--)
        {
            c[i] = p[i];
            for (int j = i + 1; j < totdeg; j++)
            {
                c[i] -= T[i][j] * c[j];
            }
        }

        // Step 6: Convert to double and return
        totdeg--;
        std::vector<std::complex<double>> result(totdeg);

        for (int i = 0; i < totdeg; i++)
        {
            double val = to_double(c[i]);
            result[i] = std::complex<double>(val, 0.0);
        }

        return result;
    }

} // namespace heongpu
