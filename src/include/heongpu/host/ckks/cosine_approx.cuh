// Copyright 2025-2026 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

// Uses NTL library for high-precision computation (similar to
// github.com/DohyeongKi/better-homomorphic-sine-evaluation)

#ifndef HEONGPU_COSINE_APPROX_H
#define HEONGPU_COSINE_APPROX_H

#include <vector>
#include <complex>

namespace heongpu
{

    // ApproximateCos computes a polynomial approximation of degree "degree" in
    // Chebyshev basis of the function cos(2*pi*x/2^"scnum") in the range -"K"
    // to "K"
    //
    // Parameters:
    //   K      - Range is [-K, K]
    //   degree - Upper bound for polynomial degree
    //   dev    - Deviation parameter (1/dev is the interval size around each
    //   integer) scnum  - Scaling parameter (approximates cos(2*pi*x/2^scnum))
    //   period - Period multiplier (approximates cos(2*pi*x/(period*2^scnum)))
    //   shift  - Node shift (approximates cos(2*pi*(x-shift)/(period*2^scnum)))
    //
    // Returns:
    //   Vector of Chebyshev coefficients [c0, c1, c2, ..., cn] where
    //   p(x) = sum(ci * Ti(x)) for normalized x in [-1, 1]
    std::vector<std::complex<double>> ApproximateCos(int K, int degree,
                                                     double dev, int scnum,
                                                     int period = 1,
                                                     double shift = 0.25);

} // namespace heongpu

#endif // HEONGPU_COSINE_APPROX_H
