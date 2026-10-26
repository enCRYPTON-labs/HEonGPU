// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <algorithm>
#include <heongpu/kernel/contextpool.hpp>

namespace heongpu
{

    std::vector<int> KeySwitchParameterGenerator::d_counter(const int l,
                                                            const int m)
    {
        int l_ = l;

        std::vector<int> result;
        while (l_ > 0)
        {
            if (l_ > m)
            {
                result.push_back(m);
                l_ = l_ - m;
            }
            else
            {
                result.push_back(l_);
                break;
            }
        }

        return result;
    }

    std::vector<int> KeySwitchParameterGenerator::d_location_counter(
        const std::vector<int> d_counter)
    {
        std::vector<int> result;
        result.push_back(0);

        for (int i = 0; i < d_counter.size() - 1; i++)
        {
            result.push_back(result[i] + d_counter[i]);
        }

        return result;
    }

    std::vector<int> KeySwitchParameterGenerator::sk_pair_counter(
        const std::vector<int> d_counter, int P_size)
    {
        std::vector<int> result;

        for (int i = 0; i < d_counter.size(); i++)
        {
            for (int j = 0; j < d_counter[i]; j++)
            {
                result.push_back(i);
            }
        }

        for (int i = 0; i < P_size; i++)
        {
            result.push_back(INT32_MAX);
        }

        return result;
    }

    KeySwitchParameterGenerator::KeySwitchParameterGenerator(
        int poly_degree, std::vector<Data64> modulus, int P_size,
        scheme_type scheme, keyswitching_type method)
    {
        switch (static_cast<int>(scheme))
        {
            case 1: // BFV
                if (static_cast<int>(method) == 2)
                {
                    n_ = poly_degree;

                    // Digits of one prime with one P prime, as for CKKS. With
                    // more P primes BFV keeps the digits of two primes that
                    // keys of earlier versions were generated with.
                    m = std::min(P_size, 2);

                    first_Qtilda_ = modulus.size();
                    first_Q_ = first_Qtilda_ - P_size;
                    first_P_ = P_size;

                    for (int i = 0; i < first_Qtilda_; i++)
                    {
                        modulus_vector.push_back((Modulus64) modulus[i]);
                    }

                    d_vector_ = d_counter(first_Q_, m);

                    d_ = d_vector_.size();
                }
                else
                {
                    throw ParameterError(error_message(
                        "Invalid Key Switching Type ", static_cast<int>(method),
                        "! The key switching context pool needs "
                        "KEYSWITCHING_METHOD_II (2)."));
                }
                break;
            case 2: // CKKS

                if (static_cast<int>(method) == 2)
                {
                    n_ = poly_degree;

                    m = P_size;

                    first_Qtilda_ = modulus.size();
                    first_Q_ = first_Qtilda_ - P_size;
                    first_P_ = P_size;

                    for (int i = 0; i < first_Qtilda_; i++)
                    {
                        modulus_vector.push_back((Modulus64) modulus[i]);
                    }

                    for (int i = 0; i < first_Q_; i++)
                    {
                        level_Q_.push_back(first_Q_ - i);
                        level_Qtilda_.push_back(first_Qtilda_ - i);
                    }

                    for (int i = 0; i < first_Q_; i++)
                    {
                        std::vector<int> d_vector_inner =
                            d_counter(level_Q_[i], m);

                        level_d_vector_.push_back(d_vector_inner);
                    }

                    for (int i = 0; i < first_Q_; i++)
                    {
                        level_d_.push_back(level_d_vector_[i].size());
                    }
                }
                else
                {
                    throw ParameterError(error_message(
                        "Invalid Key Switching Type ", static_cast<int>(method),
                        "! The key switching context pool needs "
                        "KEYSWITCHING_METHOD_II (2)."));
                }

                break;
            default:
                throw std::invalid_argument("invalid scheme type");
                break;
        }
    };

    int KeySwitchParameterGenerator::B_counter(
        const int n, const int m, const std::vector<int> dtilda_counter)
    {
        // 2 * n * d * max(D) * max(D')

        int total_bit = 1 + calculate_bit_size(n) +
                        calculate_bit_size(dtilda_counter.size()) + (m * 60) +
                        (m * 60);

        float result = static_cast<float>(total_bit);

        return int((total_bit / 60) + 1.0);
    }

    std::vector<Data64>
    KeySwitchParameterGenerator::base_change_matrix_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<Data64> base_matrix;
        int index = 0;
        for (int l = 0; l < d_; l++)
        {
            for (int k = 0; k < obase.size(); k++)
            {
                for (int i = 0; i < d_vector_[l]; i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < d_vector_[l]; j++)
                    {
                        if (i != j)
                        {
                            temp = OPERATOR64::mult(
                                temp, ibase[j + index].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            index = index + d_vector_[l];
        }

        return base_matrix;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_base_change_matrix_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = modulus_vector;

        int elementToRemove = first_Q_ - 1;

        std::vector<std::vector<Data64>> all_base_matrix;
        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> all_base_matrix_inner;
            int index = 0;
            for (int l = 0; l < level_d_[main_lp]; l++)
            {
                for (int k = 0; k < obase.size(); k++)
                {
                    for (int i = 0; i < level_d_vector_[main_lp][l]; i++)
                    {
                        Data64 temp = 1;
                        for (int j = 0; j < level_d_vector_[main_lp][l]; j++)
                        {
                            if (i != j)
                            {
                                Data64 base_inner =
                                    ibase[j + index].value % obase[k].value;
                                temp = OPERATOR64::mult(temp, base_inner,
                                                        obase[k]);
                            }
                        }
                        all_base_matrix_inner.push_back(temp);
                    }
                }
                index = index + level_d_vector_[main_lp][l];
            }

            ibase.erase(ibase.begin() + elementToRemove);
            obase.erase(obase.begin() + elementToRemove);
            elementToRemove--;

            all_base_matrix.push_back(all_base_matrix_inner);
        }

        return all_base_matrix;
    }

    std::vector<Data64> KeySwitchParameterGenerator::Mi_inv_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Data64> inv_Mi;

        int index = 0;
        for (int l = 0; l < d_; l++)
        {
            for (int i = 0; i < d_vector_[l]; i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < d_vector_[l]; j++)
                {
                    if (i != j)
                    {
                        temp = OPERATOR64::mult(temp, ibase[j + index].value,
                                                ibase[i + index]);
                    }
                }
                inv_Mi.push_back(OPERATOR64::modinv(temp, ibase[i + index]));
            }

            index = index + d_vector_[l];
        }

        return inv_Mi;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_Mi_inv_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;

        int elementToRemove = first_Q_ - 1;

        std::vector<std::vector<Data64>> all_inv_Mi;
        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> inv_Mi;

            int index = 0;
            for (int l = 0; l < level_d_[main_lp]; l++)
            {
                for (int i = 0; i < level_d_vector_[main_lp][l]; i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < level_d_vector_[main_lp][l]; j++)
                    {
                        if (i != j)
                        {
                            Data64 ibase_inner =
                                ibase[j + index].value % ibase[i + index].value;
                            temp = OPERATOR64::mult(temp, ibase_inner,
                                                    ibase[i + index]);
                        }
                    }
                    inv_Mi.push_back(
                        OPERATOR64::modinv(temp, ibase[i + index]));
                }

                index = index + level_d_vector_[main_lp][l];
            }

            ibase.erase(ibase.begin() + elementToRemove);
            elementToRemove--;

            all_inv_Mi.push_back(inv_Mi);
        }

        return all_inv_Mi;
    }

    std::vector<int> KeySwitchParameterGenerator::I_j()
    {
        return d_vector_;
    }

    std::vector<int> KeySwitchParameterGenerator::I_j_2()
    {
        return dtilda_vector_;
    }

    std::vector<std::vector<int>> KeySwitchParameterGenerator::level_I_j()
    {
        return level_d_vector_;
    }

    std::vector<std::vector<int>> KeySwitchParameterGenerator::level_I_j_2()
    {
        return level_dtilda_vector_;
    }

    std::vector<int> KeySwitchParameterGenerator::I_location()
    {
        return d_location_counter(d_vector_);
    }

    std::vector<int> KeySwitchParameterGenerator::I_location_2()
    {
        return d_location_counter(dtilda_vector_);
    }

    std::vector<std::vector<int>>
    KeySwitchParameterGenerator::level_I_location()
    {
        std::vector<std::vector<int>> all_dtilda_location;

        for (int i = 0; i < level_d_vector_.size(); i++)
        {
            all_dtilda_location.push_back(
                d_location_counter(level_d_vector_[i]));
        }

        return all_dtilda_location;
    }

    std::vector<std::vector<int>>
    KeySwitchParameterGenerator::level_I_location_2()
    {
        std::vector<std::vector<int>> all_dtilda_location;

        for (int i = 0; i < level_dtilda_vector_.size(); i++)
        {
            all_dtilda_location.push_back(
                d_location_counter(level_dtilda_vector_[i]));
        }

        return all_dtilda_location;
    }

    std::vector<Data64> KeySwitchParameterGenerator::prod_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<int> I_j_ = KeySwitchParameterGenerator::I_j();
        std::vector<int> I_location_ =
            KeySwitchParameterGenerator::I_location();

        std::vector<Data64> prod;

        for (int l = 0; l < d_; l++)
        {
            for (int i = 0; i < obase.size(); i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < I_j_[l]; j++)
                {
                    temp = OPERATOR64::mult(
                        temp, ibase[j + I_location_[l]].value, obase[i]);
                }
                prod.push_back(temp);
            }
        }

        return prod;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_prod_D_to_Qtilda()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<std::vector<int>> I_j_ =
            KeySwitchParameterGenerator::level_I_j();
        std::vector<std::vector<int>> I_location_ =
            KeySwitchParameterGenerator::level_I_location();

        std::vector<std::vector<Data64>> all_prod;
        int elementToRemove = first_Q_ - 1;

        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> prod;

            for (int l = 0; l < level_d_[main_lp]; l++)
            {
                for (int i = 0; i < obase.size(); i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < I_j_[main_lp][l]; j++)
                    {
                        Data64 ibase_inner =
                            ibase[j + I_location_[main_lp][l]].value %
                            obase[i].value;
                        temp = OPERATOR64::mult(temp, ibase_inner, obase[i]);
                    }
                    prod.push_back(temp);
                }
            }

            ibase.erase(ibase.begin() + elementToRemove);
            obase.erase(obase.begin() + elementToRemove);
            elementToRemove--;

            all_prod.push_back(prod);
        }

        return all_prod;
    }

    std::vector<int> KeySwitchParameterGenerator::sk_pair()
    {
        return sk_pair_counter(d_vector_, first_P_);
    }

    std::vector<std::vector<int>> KeySwitchParameterGenerator::level_sk_pair()
    {
        std::vector<std::vector<int>> all_sk_pair_new;

        for (int i = 0; i < level_d_vector_.size(); i++)
        {
            all_sk_pair_new.push_back(
                sk_pair_counter(level_d_vector_[i], first_P_));
        }

        return all_sk_pair_new;
    }

    std::vector<Root64> KeySwitchParameterGenerator::B_prime_ntt_tables()
    {
        int lg = log2(n_);
        std::vector<Root64> forward_table; // bit reverse order

        for (int i = 0; i < r_prime_; i++)
        {
            std::vector<Root64> table;
            table.push_back(1);

            for (int j = 1; j < n_; j++)
            {
                Data64 exp = OPERATOR64::mult(table[(j - 1)], B_prime_psi[i],
                                              B_prime[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                forward_table.push_back(table[gpuntt::bitreverse(j, lg)]);
            }
        }

        return forward_table;
    }

    std::vector<Root64> KeySwitchParameterGenerator::B_prime_intt_tables()
    {
        int lg = log2(n_);
        std::vector<Root64> forward_table; // bit reverse order

        for (int i = 0; i < r_prime_; i++)
        {
            std::vector<Root64> table;
            table.push_back(1);
            Data64 inv_root = OPERATOR64::modinv(B_prime_psi[i], B_prime[i]);
            for (int j = 1; j < n_; j++)
            {
                Data64 exp =
                    OPERATOR64::mult(table[(j - 1)], inv_root, B_prime[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                forward_table.push_back(table[gpuntt::bitreverse(j, lg)]);
            }
        }

        return forward_table;
    }

    std::vector<Ninverse64> KeySwitchParameterGenerator::B_prime_n_inverse()
    {
        Data64 n_inner = n_;
        std::vector<Ninverse64> n_inverse_;
        for (int i = 0; i < B_prime.size(); i++)
        {
            n_inverse_.push_back(OPERATOR64::modinv(n_inner, B_prime[i]));
        }

        return n_inverse_;
    }

    std::vector<Data64> KeySwitchParameterGenerator::base_change_matrix_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = B_prime;

        std::vector<Data64> base_matrix;
        int index = 0;
        for (int l = 0; l < d_tilda_; l++)
        {
            for (int k = 0; k < obase.size(); k++)
            {
                for (int i = 0; i < dtilda_vector_[l]; i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < dtilda_vector_[l]; j++)
                    {
                        if (i != j)
                        {
                            temp = OPERATOR64::mult(
                                temp, ibase[j + index].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            index = index + dtilda_vector_[l];
        }

        return base_matrix;
    }

    std::vector<Data64> KeySwitchParameterGenerator::base_change_matrix_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<Data64> base_matrix;
        for (int k = 0; k < obase.size(); k++)
        {
            for (int i = 0; i < ibase.size(); i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < ibase.size(); j++)
                {
                    if (i != j)
                    {
                        Data64 B_ = ibase[j].value % obase[k].value;
                        temp = OPERATOR64::mult(temp, B_, obase[k]);
                    }
                }
                base_matrix.push_back(temp);
            }
        }

        return base_matrix;
    }

    std::vector<Data64> KeySwitchParameterGenerator::Mi_inv_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Data64> inv_Mi;

        int index = 0;
        for (int l = 0; l < d_tilda_; l++)
        {
            for (int i = 0; i < dtilda_vector_[l]; i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < dtilda_vector_[l]; j++)
                {
                    if (i != j)
                    {
                        temp = OPERATOR64::mult(temp, ibase[j + index].value,
                                                ibase[i + index]);
                    }
                }
                inv_Mi.push_back(OPERATOR64::modinv(temp, ibase[i + index]));
            }

            index = index + dtilda_vector_[l];
        }

        return inv_Mi;
    }

    std::vector<Data64> KeySwitchParameterGenerator::Mi_inv_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Data64> inv_Mi;

        for (int l = 0; l < d_tilda_; l++)
        {
            for (int i = 0; i < ibase.size(); i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < ibase.size(); j++)
                {
                    if (i != j)
                    {
                        temp = OPERATOR64::mult(temp, ibase[j].value, ibase[i]);
                    }
                }
                inv_Mi.push_back(OPERATOR64::modinv(temp, ibase[i]));
            }
        }

        return inv_Mi;
    }

    std::vector<Data64> KeySwitchParameterGenerator::prod_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = B_prime;

        std::vector<int> I_j_ = KeySwitchParameterGenerator::I_j_2();
        std::vector<int> I_location_ =
            KeySwitchParameterGenerator::I_location_2();

        std::vector<Data64> prod;

        for (int l = 0; l < d_tilda_; l++)
        { // dtilda
            for (int i = 0; i < r_prime_; i++)
            { // r_prime
                Data64 temp = 1;
                for (int j = 0; j < I_j_[l]; j++)
                {
                    temp = OPERATOR64::mult(
                        temp, ibase[j + I_location_[l]].value, obase[i]);
                }
                prod.push_back(temp);
            }
        }

        return prod;
    }

    std::vector<Data64> KeySwitchParameterGenerator::prod_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<int> I_j_ = KeySwitchParameterGenerator::I_j_2();
        std::vector<int> I_location_ =
            KeySwitchParameterGenerator::I_location_2();

        std::vector<Data64> prod;

        for (int l = 0; l < d_tilda_; l++)
        { // dtilda
            for (int i = 0; i < I_j_[l]; i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < r_prime_; j++)
                { // r_prime
                    Data64 base =
                        ibase[j].value % obase[i + I_location_[l]].value;
                    temp =
                        OPERATOR64::mult(temp, base, obase[i + I_location_[l]]);
                }
                prod.push_back(temp);
            }
        }

        return prod;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_base_change_matrix_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = B_prime;

        int elementToRemove = first_Q_ - 1;

        std::vector<std::vector<Data64>> all_base_matrix;
        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> all_base_matrix_inner;
            int index = 0;
            for (int l = 0; l < level_d_tilda_[main_lp]; l++)
            {
                for (int k = 0; k < r_prime_; k++)
                {
                    for (int i = 0; i < level_dtilda_vector_[main_lp][l]; i++)
                    {
                        Data64 temp = 1;
                        for (int j = 0; j < level_dtilda_vector_[main_lp][l];
                             j++)
                        {
                            if (i != j)
                            {
                                Data64 base_inner =
                                    ibase[j + index].value % obase[k].value;
                                temp = OPERATOR64::mult(temp, base_inner,
                                                        obase[k]);
                            }
                        }
                        all_base_matrix_inner.push_back(temp);
                    }
                }
                index = index + level_dtilda_vector_[main_lp][l];
            }

            ibase.erase(ibase.begin() + elementToRemove);
            elementToRemove--;

            all_base_matrix.push_back(all_base_matrix_inner);
        }

        return all_base_matrix;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_base_change_matrix_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Modulus64> obase = modulus_vector;

        int elementToRemove = first_Q_ - 1;

        std::vector<std::vector<Data64>> all_base_matrix;
        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> base_matrix;
            for (int k = 0; k < obase.size(); k++)
            {
                for (int i = 0; i < ibase.size(); i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < ibase.size(); j++)
                    {
                        if (i != j)
                        {
                            Data64 B_ = ibase[j].value % obase[k].value;
                            temp = OPERATOR64::mult(temp, B_, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }

            obase.erase(obase.begin() + elementToRemove);
            elementToRemove--;

            all_base_matrix.push_back(base_matrix);
        }

        return all_base_matrix;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_Mi_inv_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;

        int elementToRemove = first_Q_ - 1;

        std::vector<std::vector<Data64>> all_inv_Mi;
        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> inv_Mi;

            int index = 0;
            for (int l = 0; l < level_d_tilda_[main_lp]; l++)
            {
                for (int i = 0; i < level_dtilda_vector_[main_lp][l]; i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < level_dtilda_vector_[main_lp][l]; j++)
                    {
                        if (i != j)
                        {
                            Data64 ibase_inner =
                                ibase[j + index].value % ibase[i + index].value;
                            temp = OPERATOR64::mult(temp, ibase_inner,
                                                    ibase[i + index]);
                        }
                    }
                    inv_Mi.push_back(
                        OPERATOR64::modinv(temp, ibase[i + index]));
                }

                index = index + level_dtilda_vector_[main_lp][l];
            }

            ibase.erase(ibase.begin() + elementToRemove);
            elementToRemove--;

            all_inv_Mi.push_back(inv_Mi);
        }

        return all_inv_Mi;
    }

    std::vector<Data64> KeySwitchParameterGenerator::level_Mi_inv_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Data64> inv_Mi;

        for (int i = 0; i < ibase.size(); i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < ibase.size(); j++)
            {
                if (i != j)
                {
                    temp = OPERATOR64::mult(temp, ibase[j].value, ibase[i]);
                }
            }
            inv_Mi.push_back(OPERATOR64::modinv(temp, ibase[i]));
        }

        return inv_Mi;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_prod_D_to_B()
    {
        std::vector<Modulus64> ibase = modulus_vector;
        std::vector<Modulus64> obase = B_prime;

        std::vector<std::vector<int>> I_j_ =
            KeySwitchParameterGenerator::level_I_j_2();
        std::vector<std::vector<int>> I_location_ =
            KeySwitchParameterGenerator::level_I_location_2();

        std::vector<std::vector<Data64>> all_prod;
        int elementToRemove = first_Q_ - 1;

        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> prod;

            for (int l = 0; l < level_d_tilda_[main_lp]; l++)
            { // dtilda
                for (int i = 0; i < r_prime_; i++)
                { // r_prime
                    Data64 temp = 1;
                    for (int j = 0; j < I_j_[main_lp][l]; j++)
                    {
                        temp = OPERATOR64::mult(
                            temp, ibase[j + I_location_[main_lp][l]].value,
                            obase[i]);
                    }
                    prod.push_back(temp);
                }
            }

            ibase.erase(ibase.begin() + elementToRemove);
            elementToRemove--;

            all_prod.push_back(prod);
        }

        return all_prod;
    }

    std::vector<std::vector<Data64>>
    KeySwitchParameterGenerator::level_prod_B_to_D()
    {
        std::vector<Modulus64> ibase = B_prime;
        std::vector<Modulus64> obase = modulus_vector;

        std::vector<std::vector<int>> I_j_ =
            KeySwitchParameterGenerator::level_I_j_2();
        std::vector<std::vector<int>> I_location_ =
            KeySwitchParameterGenerator::level_I_location_2();

        std::vector<std::vector<Data64>> all_prod;
        int elementToRemove = first_Q_ - 1;

        for (int main_lp = 0; main_lp < level_Q_.size(); main_lp++)
        {
            std::vector<Data64> prod;

            for (int l = 0; l < level_d_tilda_[main_lp]; l++)
            { // dtilda
                for (int i = 0; i < I_j_[main_lp][l]; i++)
                {
                    Data64 temp = 1;
                    for (int j = 0; j < r_prime_; j++)
                    { // r_prime
                        Data64 base = ibase[j].value %
                                      obase[i + I_location_[main_lp][l]].value;
                        temp = OPERATOR64::mult(
                            temp, base, obase[i + I_location_[main_lp][l]]);
                    }
                    prod.push_back(temp);
                }
            }

            obase.erase(obase.begin() + elementToRemove);
            elementToRemove--;

            all_prod.push_back(prod);
        }

        return all_prod;
    }

} // namespace heongpu
