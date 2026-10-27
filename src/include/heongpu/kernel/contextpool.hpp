// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_PRIMEPOOL_H
#define HEONGPU_PRIMEPOOL_H

#include "gpuntt/common/nttparameters.cuh"
#include <heongpu/util/util.cuh>
#include <heongpu/util/schemes.h>

namespace heongpu
{

    class KeySwitchParameterGenerator
    {
        friend class Parameters;
        template <Scheme S> friend class HEContextImpl;

      public:
        KeySwitchParameterGenerator(int poly_degree,
                                    std::vector<Data64> modulus, int P_size,
                                    scheme_type scheme);

      private:
        int n_;
        int m = 2;

        std::vector<Modulus64> modulus_vector;
        std::vector<Modulus64> B_prime;
        std::vector<Data64> B_prime_psi;

        int B_counter(const int n, const int m,
                      const std::vector<int> dtilda_counter);

        int first_Qtilda_;
        int first_Q_;
        int first_P_;
        int d_; // bfv
        int d_tilda_;

        int r_prime_;

        std::vector<int> level_Qtilda_; // ckks
        std::vector<int> level_Q_; // ckks
        std::vector<int> level_d_; // ckks
        std::vector<int> level_d_tilda_; // ckks

        std::vector<int> d_vector_; // bfv
        std::vector<int> dtilda_vector_; // bfv
        std::vector<std::vector<int>> level_d_vector_; // ckks
        std::vector<std::vector<int>> level_dtilda_vector_; // ckks

        std::vector<Data64> base_change_matrix_D_to_Qtilda(); // bfv
        std::vector<Data64> Mi_inv_D_to_Qtilda(); // bfv

        std::vector<std::vector<Data64>>
        level_base_change_matrix_D_to_Qtilda(); // ckks
        std::vector<std::vector<Data64>> level_Mi_inv_D_to_Qtilda(); // ckks

        std::vector<int> I_j(); // bfv
        std::vector<int> I_j_2(); // bfv
        std::vector<int> I_location(); // bfv
        std::vector<int> I_location_2(); // bfv
        std::vector<int> sk_pair(); // bfv

        std::vector<std::vector<int>> level_I_j(); // ckks
        std::vector<std::vector<int>> level_I_j_2(); // ckks
        std::vector<std::vector<int>> level_I_location(); // ckks
        std::vector<std::vector<int>> level_I_location_2(); // ckks
        std::vector<std::vector<int>> level_sk_pair(); // ckks

        std::vector<Data64> prod_D_to_Qtilda(); // bfv
        std::vector<std::vector<Data64>> level_prod_D_to_Qtilda(); // ckks

        std::vector<int> d_counter(const int l, const int m);
        std::vector<int> d_location_counter(const std::vector<int> d_counter);
        std::vector<int> sk_pair_counter(const std::vector<int> d_counter,
                                         int Q_size);

        std::vector<Root64> B_prime_ntt_tables();
        std::vector<Root64> B_prime_intt_tables();
        std::vector<Ninverse64> B_prime_n_inverse();

        std::vector<Data64> base_change_matrix_D_to_B();
        std::vector<Data64> base_change_matrix_B_to_D();
        std::vector<Data64> Mi_inv_D_to_B();
        std::vector<Data64> Mi_inv_B_to_D();

        std::vector<Data64> prod_D_to_B();
        std::vector<Data64> prod_B_to_D();

        std::vector<std::vector<Data64>> level_base_change_matrix_D_to_B();
        std::vector<std::vector<Data64>> level_base_change_matrix_B_to_D();
        std::vector<std::vector<Data64>> level_Mi_inv_D_to_B();
        std::vector<Data64> level_Mi_inv_B_to_D();

        std::vector<std::vector<Data64>> level_prod_D_to_B();
        std::vector<std::vector<Data64>> level_prod_B_to_D();
    };

} // namespace heongpu
#endif // HEONGPU_PRIMEPOOL_H
