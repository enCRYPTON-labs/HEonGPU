// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/context.cuh>

namespace heongpu
{
    HEContextImpl<Scheme::BFV>::HEContextImpl(const sec_level_type sec_level)
    {
        if (!coeff_modulus_specified_)
        {
            scheme_ = scheme_type::bfv;
            sec_level_ = sec_level;
            keyswitching_type_ = keyswitching_type::NONE;
        }
        else
        {
            throw std::logic_error("Parameters cannot be changed after the "
                                   "coeff_modulus is specified!");
        }
    }

    void HEContextImpl<Scheme::BFV>::set_poly_modulus_degree(
        size_t poly_modulus_degree)
    {
        if ((!coeff_modulus_specified_) && (!poly_modulus_degree_specified_))
        {
            if (!is_power_of_two(poly_modulus_degree))
            {
                throw std::logic_error(
                    "Poly modulus degree have to be power of two");
            }

            if ((poly_modulus_degree > MAX_POLY_DEGREE) ||
                (poly_modulus_degree < MIN_POLY_DEGREE))
            {
                throw std::logic_error("Poly modulus degree is not supported");
            }

            n = poly_modulus_degree;
            n_power = int(log2l(n));

            poly_modulus_degree_specified_ = true;
        }
        else
        {
            throw std::logic_error("Poly modulus degree cannot be changed "
                                   "after the coeff_modulus is specified!");
        }
    }

    void HEContextImpl<Scheme::BFV>::set_coeff_modulus_bit_sizes(
        const std::vector<int>& log_Q_bases_bit_sizes,
        const std::vector<int>& log_P_bases_bit_sizes)
    {
        if ((!coeff_modulus_specified_) && (!context_generated_) &&
            (poly_modulus_degree_specified_))
        {
            if (log_P_bases_bit_sizes.empty())
            {
                throw std::logic_error(
                    "log_P_bases_bit_sizes cannot be empty!");
            }

            keyswitching_type_ =
                (log_P_bases_bit_sizes.size() == 1)
                    ? keyswitching_type::KEYSWITCHING_METHOD_I
                    : keyswitching_type::KEYSWITCHING_METHOD_II;

            if (!coefficient_validator(log_Q_bases_bit_sizes,
                                       log_P_bases_bit_sizes))
            {
                throw std::logic_error("P should be bigger than Q pairs!");
            }

            // Q' = Q x P
            Qprime_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            Qprime_mod_bit_sizes_.insert(Qprime_mod_bit_sizes_.end(),
                                         log_P_bases_bit_sizes.begin(),
                                         log_P_bases_bit_sizes.end());

            Q_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            P_mod_bit_sizes_ = log_P_bases_bit_sizes;

            total_coeff_bit_count = 0; // TODO: calculate it with prod
            for (int i = 0; i < Qprime_mod_bit_sizes_.size(); i++)
            {
                total_coeff_bit_count =
                    total_coeff_bit_count + Qprime_mod_bit_sizes_[i];
            }

            int max_coeff_bit_count = 0;
            switch (sec_level_)
            {
                case sec_level_type::none:
                    break;
                case sec_level_type::sec128:
                    max_coeff_bit_count = heongpu_128bit_std_parms(n);
                    break;
                case sec_level_type::sec192:
                    max_coeff_bit_count = heongpu_192bit_std_parms(n);
                    break;
                case sec_level_type::sec256:
                    max_coeff_bit_count = heongpu_256bit_std_parms(n);
                    break;
                default:
                    throw std::runtime_error(error_message(
                        "Invalid security level ", static_cast<int>(sec_level_),
                        "! Expected sec128, sec192 or sec256."));
                    break;
            }

            if ((max_coeff_bit_count < total_coeff_bit_count) &&
                (sec_level_ != sec_level_type::none))
            {
                throw std::runtime_error(
                    "Parameters do not align with the security recommendations "
                    "provided by the lattice-estimator");
            }

            // Q' bases size
            Q_prime_size = Qprime_mod_bit_sizes_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = log_Q_bases_bit_sizes.size();

            // P bases size
            P_size = Q_prime_size - Q_size;

            prime_vector_ =
                generate_primes(n,
                                Qprime_mod_bit_sizes_); // prime_vector_

            for (int i = 0; i < prime_vector_.size(); i++)
            {
                base_q.push_back(prime_vector_[i].value);
            }

            coeff_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("Coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContextImpl<Scheme::BFV>::set_coeff_modulus_values(
        const std::vector<Data64>& log_Q_bases,
        const std::vector<Data64>& log_P_bases)
    {
        if ((!coeff_modulus_specified_) && (!context_generated_) &&
            (poly_modulus_degree_specified_))
        {
            std::vector<int> log_Q_bases_bit_sizes;
            for (int i = 0; i < log_Q_bases.size(); i++)
            {
                log_Q_bases_bit_sizes.push_back(
                    calculate_bit_size(log_Q_bases[i]));
            }

            std::vector<int> log_P_bases_bit_sizes;
            for (int i = 0; i < log_P_bases.size(); i++)
            {
                log_P_bases_bit_sizes.push_back(
                    calculate_bit_size(log_P_bases[i]));
            }

            if (log_P_bases_bit_sizes.empty())
            {
                throw std::logic_error(
                    "log_P_bases_bit_sizes cannot be empty!");
            }

            keyswitching_type_ =
                (log_P_bases_bit_sizes.size() == 1)
                    ? keyswitching_type::KEYSWITCHING_METHOD_I
                    : keyswitching_type::KEYSWITCHING_METHOD_II;

            if (!coefficient_validator(log_Q_bases_bit_sizes,
                                       log_P_bases_bit_sizes))
            {
                throw std::logic_error(
                    "Invalid parameters, P should be bigger than Q pairs!");
            }

            // Q' = Q x P
            Qprime_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            Qprime_mod_bit_sizes_.insert(Qprime_mod_bit_sizes_.end(),
                                         log_P_bases_bit_sizes.begin(),
                                         log_P_bases_bit_sizes.end());

            Q_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            P_mod_bit_sizes_ = log_P_bases_bit_sizes;

            total_coeff_bit_count = 0; // TODO: calculate it with prod
            for (int i = 0; i < Qprime_mod_bit_sizes_.size(); i++)
            {
                total_coeff_bit_count =
                    total_coeff_bit_count + Qprime_mod_bit_sizes_[i];
            }

            int max_coeff_bit_count = 0;
            switch (sec_level_)
            {
                case sec_level_type::none:
                    break;
                case sec_level_type::sec128:
                    max_coeff_bit_count = heongpu_128bit_std_parms(n);
                    break;
                case sec_level_type::sec192:
                    max_coeff_bit_count = heongpu_192bit_std_parms(n);
                    break;
                case sec_level_type::sec256:
                    max_coeff_bit_count = heongpu_256bit_std_parms(n);
                    break;
                default:
                    throw std::runtime_error(error_message(
                        "Invalid security level ", static_cast<int>(sec_level_),
                        "! Expected sec128, sec192 or sec256."));
                    break;
            }

            if ((max_coeff_bit_count < total_coeff_bit_count) &&
                (sec_level_ != sec_level_type::none))
            {
                throw std::runtime_error(
                    "Parameters do not align with the security recommendations "
                    "provided by the lattice-estimator");
            }

            // Q' bases size
            Q_prime_size = Qprime_mod_bit_sizes_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = log_Q_bases_bit_sizes.size();

            // P bases size
            P_size = Q_prime_size - Q_size;

            for (int i = 0; i < log_Q_bases.size(); i++)
            {
                Modulus64 mod_in(log_Q_bases[i]);
                prime_vector_.push_back(mod_in);
            }

            for (int i = 0; i < log_P_bases.size(); i++)
            {
                Modulus64 mod_in(log_P_bases[i]);
                prime_vector_.push_back(mod_in);
            }

            for (int i = 0; i < prime_vector_.size(); i++)
            {
                base_q.push_back(prime_vector_[i].value);
            }

            coeff_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("Coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContextImpl<Scheme::BFV>::set_coeff_modulus_default_values(
        int P_modulus_size)
    {
        if ((!coeff_modulus_specified_) && (!context_generated_) &&
            (poly_modulus_degree_specified_))
        {
            if (P_modulus_size < 1)
            {
                throw std::logic_error(
                    "P_modulus_size cannot be lower than 1!");
            }

            keyswitching_type_ =
                (P_modulus_size == 1)
                    ? keyswitching_type::KEYSWITCHING_METHOD_I
                    : keyswitching_type::KEYSWITCHING_METHOD_II;

            total_coeff_bit_count = 0;
            switch (sec_level_)
            {
                case sec_level_type::sec128:
                {
                    total_coeff_bit_count = heongpu_128bit_std_parms(n);
                    prime_vector_ =
                        defaultparams::get_128bit_sec_modulus().at(n);

                    int size = prime_vector_.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }
                }
                break;
                case sec_level_type::sec192:
                {
                    total_coeff_bit_count = heongpu_192bit_std_parms(n);
                    prime_vector_ =
                        defaultparams::get_192bit_sec_modulus().at(n);

                    int size = prime_vector_.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }
                }
                break;
                case sec_level_type::sec256:
                {
                    total_coeff_bit_count = heongpu_256bit_std_parms(n);
                    prime_vector_ =
                        defaultparams::get_256bit_sec_modulus().at(n);

                    int size = prime_vector_.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector_[i].value) + 1));
                    }
                }
                break;
                default:
                    throw std::runtime_error(error_message(
                        "Invalid security level ", static_cast<int>(sec_level_),
                        "! Expected sec128, sec192 or sec256."));
                    break;
            }

            // Q' bases size
            Q_prime_size = prime_vector_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = Q_prime_size - P_modulus_size;

            // P bases size
            P_size = P_modulus_size;

            for (int i = 0; i < prime_vector_.size(); i++)
            {
                base_q.push_back(prime_vector_[i].value);
            }

            coeff_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("Coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContextImpl<Scheme::BFV>::set_plain_modulus(const int plain_modulus)
    {
        if ((!context_generated_) && (!plain_modulus_specified_))
        {
            plain_modulus_ = Modulus64(plain_modulus);

            plain_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("Plain modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContextImpl<Scheme::BFV>::generate()
    {
        generate(MemoryPoolConfig::Defaults());
    }

    void
    HEContextImpl<Scheme::BFV>::generate(const MemoryPoolConfig& pool_config)
    {
        if ((!context_generated_) && (poly_modulus_degree_specified_) &&
            (plain_modulus_specified_) && (coeff_modulus_specified_))
        {
            // For kernel stack size. Reserved before the memory pool so
            // that the pool does not take the memory the stack needs.
            ensure_kernel_stack_size(2048);

            // Memory pool initialization
            MemoryPool::instance().initialize(pool_config);
            MemoryPool::instance().use_memory_pool(pool_config.use_memory_pool);
            MemoryPool::instance().set_as_rmm_default(
                pool_config.set_as_rmm_default);
            cudaDeviceSynchronize();

            // DRNG initialization
            std::vector<unsigned char> generated_entropy(16); // for 128 bit
            if (1 !=
                RAND_bytes(generated_entropy.data(), generated_entropy.size()))
                throw SystemError("RAND_bytes failed! OpenSSL could not "
                                  "generate random bytes.");
            std::vector<unsigned char> generated_nonce(8); // for 128 bit
            if (1 != RAND_bytes(generated_nonce.data(), generated_nonce.size()))
                throw SystemError("RAND_bytes failed! OpenSSL could not "
                                  "generate random bytes.");
            std::vector<unsigned char> personalization_string = {};
            RandomNumberGenerator::instance().initialize(
                generated_entropy, generated_nonce, personalization_string,
                rngongpu::SecurityLevel::AES128, false);
            cudaDeviceSynchronize();

            modulus_ = std::make_shared<DeviceVector<Modulus64>>(prime_vector_);

            std::vector<Data64> base_q_psi =
                generate_primitive_root_of_unity(n, prime_vector_);
            std::vector<Root64> Qprime_ntt_table =
                generate_ntt_table(base_q_psi, prime_vector_, n_power);
            std::vector<Root64> Qprime_intt_table =
                generate_intt_table(base_q_psi, prime_vector_, n_power);
            std::vector<Ninverse64> Qprime_n_inverse =
                generate_n_inverse(n, prime_vector_);

            ntt_table_ =
                std::make_shared<DeviceVector<Root64>>(Qprime_ntt_table);

            intt_table_ =
                std::make_shared<DeviceVector<Root64>>(Qprime_intt_table);

            n_inverse_ =
                std::make_shared<DeviceVector<Ninverse64>>(Qprime_n_inverse);

            std::vector<Data64> last_q_modinv =
                calculate_last_q_modinv(prime_vector_, Q_prime_size, P_size);
            std::vector<Data64> half = calculate_half(prime_vector_, P_size);
            std::vector<Data64> half_mod =
                calculate_half_mod(prime_vector_, half, Q_prime_size, P_size);
            std::vector<Data64> factor =
                calculate_factor(prime_vector_, Q_size, P_size);

            last_q_modinv_ =
                std::make_shared<DeviceVector<Data64>>(last_q_modinv);

            half_p_ = std::make_shared<DeviceVector<Data64>>(half);

            half_mod_ = std::make_shared<DeviceVector<Data64>>(half_mod);

            factor_ = std::make_shared<DeviceVector<Data64>>(factor);

            ///////////////////////
            ///////////////////////
            ///////////////////////

            std::vector<Data64> Mi = calculate_Mi(prime_vector_, Q_size);
            std::vector<Data64> Mi_inv =
                calculate_Mi_inv(prime_vector_, Q_size);
            std::vector<Data64> upper_half_threshold =
                calculate_upper_half_threshold(prime_vector_, Q_size);
            std::vector<Data64> decryption_modulus =
                calculate_M(prime_vector_, Q_size);

            Mi_ = std::make_shared<DeviceVector<Data64>>(Mi);

            Mi_inv_ = std::make_shared<DeviceVector<Data64>>(Mi_inv);

            upper_half_threshold_ =
                std::make_shared<DeviceVector<Data64>>(upper_half_threshold);

            decryption_modulus_ =
                std::make_shared<DeviceVector<Data64>>(decryption_modulus);

            total_bit_count_ = calculate_big_integer_bit_count(
                decryption_modulus.data(), decryption_modulus.size());

            Modulus64 plain_mod = plain_modulus_;

            Data64 plain_psi = find_minimal_primitive_root(2 * n, plain_mod);

            std::vector<Root64> plain_forward_table =
                generate_ntt_table({plain_psi}, {plain_mod}, n_power);
            std::vector<Root64> plain_inverse_table =
                generate_intt_table({plain_psi}, {plain_mod}, n_power);
            Data64 n_ = n;
            std::vector<Ninverse64> plain_n_inverse;
            std::vector<Modulus64> plain_mod2;
            plain_n_inverse.push_back(OPERATOR64::modinv(n_, plain_mod));
            plain_mod2.push_back(plain_mod);

            Data64 plain_upper_half_threshold = (plain_mod.value + 1) >> 1;

            std::vector<Data64> plain_upper_half_increment;
            for (int i = 0; i < Q_size; i++)
            {
                plain_upper_half_increment.push_back(prime_vector_[i].value -
                                                     plain_mod.value);
            }

            Modulus64 m_tilde((1ULL << 32));

            Data64 Q_mod_t = generate_Q_mod_t(prime_vector_, plain_mod, Q_size);

            std::vector<Data64> coeff_div_plain_modulus =
                generate_coeff_div_plain_modulus(prime_vector_, plain_mod,
                                                 Q_size);

            bsk_modulus = prime_vector_.size();
            if (calculate_bit_count(plain_mod.value) + total_coeff_bit_count +
                    32 >=
                MAX_MOD_BIT_COUNT * Q_size + MAX_MOD_BIT_COUNT)
            {
                bsk_modulus++;
            }

            std::vector<Modulus64> base_Bsk_mod = generate_internal_primes(
                n,
                bsk_modulus + 1); // extra for gamma parameter

            Modulus64 gamma_mod = base_Bsk_mod[bsk_modulus];
            base_Bsk_mod.pop_back();

            std::vector<Data64> base_Bsk_psi =
                generate_primitive_root_of_unity(n, base_Bsk_mod);
            std::vector<Root64> Bsk_ntt_table =
                generate_ntt_table(base_Bsk_psi, base_Bsk_mod, n_power);
            std::vector<Root64> Bsk_intt_table =
                generate_intt_table(base_Bsk_psi, base_Bsk_mod, n_power);
            std::vector<Ninverse64> Bsk_n_inverse =
                generate_n_inverse(n, base_Bsk_mod);

            base_Bsk_ = std::make_shared<DeviceVector<Modulus64>>(base_Bsk_mod);

            bsk_ntt_tables_ =
                std::make_shared<DeviceVector<Root64>>(Bsk_ntt_table);

            bsk_intt_tables_ =
                std::make_shared<DeviceVector<Root64>>(Bsk_intt_table);

            bsk_n_inverse_ =
                std::make_shared<DeviceVector<Ninverse64>>(Bsk_n_inverse);

            std::vector<Data64> base_matrix_q_Bsk =
                generate_base_matrix_q_Bsk(prime_vector_, base_Bsk_mod, Q_size);

            std::vector<Data64> inv_punctured_prod_mod_base_array =
                calculate_Mi_inv(prime_vector_, Q_size);

            std::vector<Data64> base_change_matrix_m_tilde =
                generate_base_change_matrix_m_tilde(prime_vector_, m_tilde,
                                                    Q_size);

            Data64 inv_prod_q_mod_m_tilde =
                generate_inv_prod_q_mod_m_tilde(prime_vector_, m_tilde, Q_size);

            std::vector<Data64> inv_m_tilde_mod_Bsk =
                generate_inv_m_tilde_mod_Bsk(base_Bsk_mod, m_tilde);

            std::vector<Data64> prod_q_mod_Bsk =
                generate_prod_q_mod_Bsk(prime_vector_, base_Bsk_mod, Q_size);

            std::vector<Data64> inv_prod_q_mod_Bsk =
                generate_inv_prod_q_mod_Bsk(prime_vector_, base_Bsk_mod,
                                            Q_size);

            std::vector<Data64> base_matrix_Bsk_q =
                generate_base_matrix_Bsk_q(prime_vector_, base_Bsk_mod, Q_size);

            std::vector<Data64> base_change_matrix_msk =
                generate_base_change_matrix_msk(base_Bsk_mod);

            std::vector<Data64> inv_punctured_prod_mod_B_array =
                generate_inv_punctured_prod_mod_B_array(base_Bsk_mod);

            Data64 inv_prod_B_mod_m_sk =
                generate_inv_prod_B_mod_m_sk(base_Bsk_mod);

            std::vector<Data64> prod_B_mod_q =
                generate_prod_B_mod_q(prime_vector_, base_Bsk_mod, Q_size);

            std::vector<Modulus64> q_Bsk_merge_modulus =
                generate_q_Bsk_merge_modulus(prime_vector_, base_Bsk_mod,
                                             Q_size);

            std::vector<Data64> q_Bsk_merge_root =
                generate_q_Bsk_merge_root(base_q_psi, base_Bsk_psi, Q_size);

            std::vector<Root64> q_Bsk_forward_tables = generate_ntt_table(
                q_Bsk_merge_root, q_Bsk_merge_modulus, n_power);
            std::vector<Root64> q_Bsk_inverse_tables = generate_intt_table(
                q_Bsk_merge_root, q_Bsk_merge_modulus, n_power);
            std::vector<Ninverse64> q_Bsk_n_inverse =
                generate_n_inverse(n, q_Bsk_merge_modulus);

            std::vector<Data64> Qi_t =
                generate_Qi_t(prime_vector_, plain_mod, Q_size);

            std::vector<Data64> Qi_gamma =
                generate_Qi_gamma(prime_vector_, gamma_mod, Q_size);

            std::vector<Data64> Qi_inverse =
                generate_Qi_inverse(prime_vector_, Q_size);

            Data64 mulq_inv_t =
                generate_mulq_inv_t(prime_vector_, plain_mod, Q_size);

            Data64 mulq_inv_gamma =
                generate_mulq_inv_gamma(prime_vector_, gamma_mod, Q_size);

            Data64 inv_gamma = generate_inv_gamma(plain_mod, gamma_mod);

            m_tilde_ = m_tilde;

            base_change_matrix_Bsk_ =
                std::make_shared<DeviceVector<Data64>>(base_matrix_q_Bsk);

            inv_punctured_prod_mod_base_array_ =
                std::make_shared<DeviceVector<Data64>>(
                    inv_punctured_prod_mod_base_array);

            base_change_matrix_m_tilde_ =
                std::make_shared<DeviceVector<Data64>>(
                    base_change_matrix_m_tilde);

            inv_prod_q_mod_m_tilde_ = inv_prod_q_mod_m_tilde;

            inv_m_tilde_mod_Bsk_ =
                std::make_shared<DeviceVector<Data64>>(inv_m_tilde_mod_Bsk);

            prod_q_mod_Bsk_ =
                std::make_shared<DeviceVector<Data64>>(prod_q_mod_Bsk);

            inv_prod_q_mod_Bsk_ =
                std::make_shared<DeviceVector<Data64>>(inv_prod_q_mod_Bsk);

            base_change_matrix_q_ =
                std::make_shared<DeviceVector<Data64>>(base_matrix_Bsk_q);

            base_change_matrix_msk_ =
                std::make_shared<DeviceVector<Data64>>(base_change_matrix_msk);

            inv_punctured_prod_mod_B_array_ =
                std::make_shared<DeviceVector<Data64>>(
                    inv_punctured_prod_mod_B_array);

            inv_prod_B_mod_m_sk_ = inv_prod_B_mod_m_sk;

            prod_B_mod_q_ =
                std::make_shared<DeviceVector<Data64>>(prod_B_mod_q);

            q_Bsk_merge_modulus_ =
                std::make_shared<DeviceVector<Modulus64>>(q_Bsk_merge_modulus);

            q_Bsk_merge_ntt_tables_ =
                std::make_shared<DeviceVector<Root64>>(q_Bsk_forward_tables);

            q_Bsk_merge_intt_tables_ =
                std::make_shared<DeviceVector<Root64>>(q_Bsk_inverse_tables);

            q_Bsk_n_inverse_ =
                std::make_shared<DeviceVector<Ninverse64>>(q_Bsk_n_inverse);

            plain_modulus2_ =
                std::make_shared<DeviceVector<Modulus64>>(plain_mod2);

            n_plain_inverse_ =
                std::make_shared<DeviceVector<Ninverse64>>(plain_n_inverse);

            plain_ntt_tables_ =
                std::make_shared<DeviceVector<Root64>>(plain_forward_table);

            plain_intt_tables_ =
                std::make_shared<DeviceVector<Root64>>(plain_inverse_table);

            gamma_ = gamma_mod;

            coeeff_div_plainmod_ =
                std::make_shared<DeviceVector<Data64>>(coeff_div_plain_modulus);

            Q_mod_t_ = Q_mod_t;

            upper_threshold_ = plain_upper_half_threshold;

            upper_halfincrement_ = std::make_shared<DeviceVector<Data64>>(
                plain_upper_half_increment);

            Qi_t_ = std::make_shared<DeviceVector<Data64>>(Qi_t);

            Qi_gamma_ = std::make_shared<DeviceVector<Data64>>(Qi_gamma);

            Qi_inverse_ = std::make_shared<DeviceVector<Data64>>(Qi_inverse);

            mulq_inv_t_ = mulq_inv_t;
            mulq_inv_gamma_ = mulq_inv_gamma;
            inv_gamma_ = inv_gamma;

            //////////////////////////

            switch (static_cast<int>(keyswitching_type_))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    // Deafult
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                {
                    KeySwitchParameterGenerator pool(n, base_q, P_size, scheme_,
                                                     keyswitching_type_);

                    m = pool.m;
                    l = pool.first_Q_;
                    l_tilda = pool.first_Qtilda_;

                    d = pool.d_;

                    std::vector<Data64> base_change_matrix_D_to_Q_tilda_inner =
                        pool.base_change_matrix_D_to_Qtilda();
                    base_change_matrix_D_to_Q_tilda_ =
                        std::make_shared<DeviceVector<Data64>>(
                            base_change_matrix_D_to_Q_tilda_inner);

                    std::vector<Data64> Mi_inv_D_to_Q_tilda_inner =
                        pool.Mi_inv_D_to_Qtilda();
                    Mi_inv_D_to_Q_tilda_ =
                        std::make_shared<DeviceVector<Data64>>(
                            Mi_inv_D_to_Q_tilda_inner);

                    std::vector<Data64> prod_D_to_Q_tilda_inner =
                        pool.prod_D_to_Qtilda();
                    prod_D_to_Q_tilda_ = std::make_shared<DeviceVector<Data64>>(
                        prod_D_to_Q_tilda_inner);

                    std::vector<int> I_j_inner = pool.I_j();
                    I_j_ = std::make_shared<DeviceVector<int>>(I_j_inner);

                    std::vector<int> I_location_inner = pool.I_location();
                    I_location_ =
                        std::make_shared<DeviceVector<int>>(I_location_inner);

                    std::vector<int> Sk_pair_inner = pool.sk_pair();
                    Sk_pair_ =
                        std::make_shared<DeviceVector<int>>(Sk_pair_inner);
                }
                break;
                default:
                    throw ParameterError(
                        error_message("Invalid Key Switching Type ",
                                      static_cast<int>(keyswitching_type_),
                                      "! Expected KEYSWITCHING_METHOD_I (1) or "
                                      "KEYSWITCHING_METHOD_II (2)."));
                    break;
            }

            context_id_ = context_identity(static_cast<int>(scheme_), n,
                                           prime_vector_, plain_modulus_.value);
            context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Context is already generated!");
        }
    }

    void HEContextImpl<Scheme::BFV>::print_parameters(std::ostream& os) const
    {
        if (context_generated_)
        {
            std::string scheme_string = "BFV";
            os << "==== HEonGPU a GPU Based Homomorphic Encryption "
                  "Library ====\n"
               << std::endl;
            os << "Encryption parameters:" << std::endl;
            os << "-->   scheme: " << scheme_string << std::endl;
            os << "-->   poly_modulus_degree: " << n << std::endl;
            os << "-->   Q_tilda size: Q( ";

            for (std::size_t i = 0; i < Q_mod_bit_sizes_.size() - 1; i++)
            {
                os << Q_mod_bit_sizes_[i] << " + ";
            }
            os << Q_mod_bit_sizes_.back();
            os << " ) + P( ";
            for (std::size_t i = 0; i < P_mod_bit_sizes_.size() - 1; i++)
            {
                os << P_mod_bit_sizes_[i] << " + ";
            }
            os << P_mod_bit_sizes_.back();
            os << " ) bits" << std::endl;

            os << "-->   plain_modulus: " << plain_modulus_.value << std::endl;

            os << std::endl;
        }
        else
        {
            os << "Parameters is not generated yet!" << std::endl;
        }
    }

    void HEContextImpl<Scheme::BFV>::save(std::ostream& os,
                                          serialization_format format) const
    {
        if ((poly_modulus_degree_specified_) && (coeff_modulus_specified_) &&
            (plain_modulus_specified_))
        {
            detail::write_serialization_header(
                os, format, detail::serialized_object::context, scheme_,
                context_id_);

            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &sec_level_, sizeof(sec_level_));

            os.write((char*) &keyswitching_type_, sizeof(keyswitching_type_));

            os.write((char*) &n, sizeof(n));

            os.write((char*) &n_power, sizeof(n_power));

            os.write((char*) &coeff_modulus, sizeof(coeff_modulus));

            os.write((char*) &total_coeff_bit_count,
                     sizeof(total_coeff_bit_count));

            os.write((char*) &Q_prime_size, sizeof(Q_prime_size));

            os.write((char*) &Q_size, sizeof(Q_size));

            os.write((char*) &P_size, sizeof(P_size));

            uint32_t prime_vector_count = prime_vector_.size();
            os.write((char*) &prime_vector_count, sizeof(prime_vector_count));
            os.write((char*) prime_vector_.data(),
                     sizeof(Modulus64) * prime_vector_count);

            uint32_t base_q_count = base_q.size();
            os.write((char*) &base_q_count, sizeof(base_q_count));
            os.write((char*) base_q.data(), sizeof(Data64) * base_q_count);

            uint32_t Qprime_mod_bit_sizes_count = Qprime_mod_bit_sizes_.size();
            os.write((char*) &Qprime_mod_bit_sizes_count,
                     sizeof(Qprime_mod_bit_sizes_count));
            os.write((char*) Qprime_mod_bit_sizes_.data(),
                     sizeof(int) * Qprime_mod_bit_sizes_count);

            uint32_t Q_mod_bit_sizes_count = Q_mod_bit_sizes_.size();
            os.write((char*) &Q_mod_bit_sizes_count,
                     sizeof(Q_mod_bit_sizes_count));
            os.write((char*) Q_mod_bit_sizes_.data(),
                     sizeof(int) * Q_mod_bit_sizes_count);

            uint32_t P_mod_bit_sizes_count = P_mod_bit_sizes_.size();
            os.write((char*) &P_mod_bit_sizes_count,
                     sizeof(P_mod_bit_sizes_count));
            os.write((char*) P_mod_bit_sizes_.data(),
                     sizeof(int) * P_mod_bit_sizes_count);

            os.write((char*) &plain_modulus_, sizeof(plain_modulus_));
        }
        else
        {
            throw std::runtime_error(
                "Context has no enough parameters to serialize!");
        }
    }

    void HEContextImpl<Scheme::BFV>::load(std::istream& is)
    {
        if ((!context_generated_))
        {
            const std::uint64_t loaded_context_id =
                detail::read_serialization_header(
                    is, detail::serialized_object::context, scheme_type::bfv);
            detail::validate_loaded_context_identity(
                context_id_, loaded_context_id, "HEContext");

            is.read((char*) &scheme_, sizeof(scheme_));

            is.read((char*) &sec_level_, sizeof(sec_level_));

            is.read((char*) &keyswitching_type_, sizeof(keyswitching_type_));

            is.read((char*) &n, sizeof(n));

            is.read((char*) &n_power, sizeof(n_power));

            is.read((char*) &coeff_modulus, sizeof(coeff_modulus));

            is.read((char*) &total_coeff_bit_count,
                    sizeof(total_coeff_bit_count));

            is.read((char*) &Q_prime_size, sizeof(Q_prime_size));

            is.read((char*) &Q_size, sizeof(Q_size));

            is.read((char*) &P_size, sizeof(P_size));

            uint32_t prime_vector_count;
            is.read((char*) &prime_vector_count, sizeof(prime_vector_count));
            prime_vector_.resize(prime_vector_count);
            is.read((char*) prime_vector_.data(),
                    sizeof(Modulus64) * prime_vector_count);

            uint32_t base_q_count;
            is.read((char*) &base_q_count, sizeof(base_q_count));
            base_q.resize(base_q_count);
            is.read((char*) base_q.data(), sizeof(Data64) * base_q_count);

            uint32_t Qprime_mod_bit_sizes_count;
            is.read((char*) &Qprime_mod_bit_sizes_count,
                    sizeof(Qprime_mod_bit_sizes_count));
            Qprime_mod_bit_sizes_.resize(Qprime_mod_bit_sizes_count);
            is.read((char*) Qprime_mod_bit_sizes_.data(),
                    sizeof(int) * Qprime_mod_bit_sizes_count);

            uint32_t Q_mod_bit_sizes_count;
            is.read((char*) &Q_mod_bit_sizes_count,
                    sizeof(Q_mod_bit_sizes_count));
            Q_mod_bit_sizes_.resize(Q_mod_bit_sizes_count);
            is.read((char*) Q_mod_bit_sizes_.data(),
                    sizeof(int) * Q_mod_bit_sizes_count);

            uint32_t P_mod_bit_sizes_count;
            is.read((char*) &P_mod_bit_sizes_count,
                    sizeof(P_mod_bit_sizes_count));
            P_mod_bit_sizes_.resize(P_mod_bit_sizes_count);
            is.read((char*) P_mod_bit_sizes_.data(),
                    sizeof(int) * P_mod_bit_sizes_count);

            is.read((char*) &plain_modulus_, sizeof(plain_modulus_));

            poly_modulus_degree_specified_ = true;
            coeff_modulus_specified_ = true;
            plain_modulus_specified_ = true;
            context_generated_ = false;

            this->generate();

            if (loaded_context_id != 0 && loaded_context_id != context_id_)
            {
                throw SerializationError(
                    "Corrupted HEContext binary! The parameters "
                    "do not match the context identity stored "
                    "with them.");
            }
        }
        else
        {
            throw std::runtime_error("Context has been already exist!");
        }
    }

    Data64
    HEContextImpl<Scheme::BFV>::generate_Q_mod_t(std::vector<Modulus64> primes,
                                                 Modulus64& plain_mod, int size)
    {
        Data64 result = 1;
        for (int i = 0; i < size; i++)
        {
            Data64 inner_prime = primes[i].value % plain_mod.value;
            result = OPERATOR64::mult(result, inner_prime, plain_mod);
        }

        return result;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_coeff_div_plain_modulus(
        std::vector<Modulus64> primes, Modulus64& plain_mod, int size)
    {
        std::vector<Data64> result_M(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, primes[i].value);
        }

        mpz_div_ui(result, result, plain_mod.value);

        mpz_t result_mod;
        mpz_init(result_mod);

        for (int i = 0; i < size; i++)
        {
            mpz_mod_ui(result_mod, result, primes[i].value);

            size_t mul_size;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mpz_export(
                NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result_mod));

            result_M[i] = ptr[0];
            free(ptr);
        }

        mpz_clear(result);
        mpz_clear(result_mod);

        return result_M;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_base_matrix_q_Bsk(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Data64> base_matrix_q_Bsk;
        for (int k = 0; k < bsk_mod.size(); k++)
        {
            for (int i = 0; i < size; i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < size; j++)
                {
                    if (i != j)
                    {
                        temp =
                            OPERATOR64::mult(temp, primes[j].value, bsk_mod[k]);
                    }
                }
                base_matrix_q_Bsk.push_back(temp);
            }
        }

        return base_matrix_q_Bsk;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_base_change_matrix_m_tilde(
        std::vector<Modulus64> primes, Modulus64 mtilda, int size)
    {
        std::vector<Data64> base_change_matrix_m_tilde;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data64 base = primes[j].value % mtilda.value;
                    temp = OPERATOR64::mult(temp, base, mtilda);
                }
            }
            base_change_matrix_m_tilde.push_back(temp);
        }

        return base_change_matrix_m_tilde;
    }

    Data64 HEContextImpl<Scheme::BFV>::generate_inv_prod_q_mod_m_tilde(
        std::vector<Modulus64> primes, Modulus64 mtilda, int size)
    {
        Data64 inv_prod_q_mod_m_tilde = 1;
        for (int i = 0; i < size; i++)
        {
            Data64 base = primes[i].value % mtilda.value;
            inv_prod_q_mod_m_tilde =
                OPERATOR64::mult(inv_prod_q_mod_m_tilde, base, mtilda);
        }

        inv_prod_q_mod_m_tilde =
            modInverse(inv_prod_q_mod_m_tilde, mtilda.value);

        return inv_prod_q_mod_m_tilde;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_inv_m_tilde_mod_Bsk(
        std::vector<Modulus64> bsk_mod, Modulus64 mtilda)
    {
        std::vector<Data64> inv_m_tilde_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            inv_m_tilde_mod_Bsk.push_back(
                OPERATOR64::modinv(mtilda.value, bsk_mod[i]));
        }

        return inv_m_tilde_mod_Bsk;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_prod_q_mod_Bsk(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Data64> prod_q_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                temp = OPERATOR64::mult(temp, primes[j].value, bsk_mod[i]);
            }
            prod_q_mod_Bsk.push_back(temp);
        }

        return prod_q_mod_Bsk;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_inv_prod_q_mod_Bsk(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Data64> inv_prod_q_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                temp = OPERATOR64::mult(temp, primes[j].value, bsk_mod[i]);
            }
            inv_prod_q_mod_Bsk.push_back(OPERATOR64::modinv(temp, bsk_mod[i]));
        }

        return inv_prod_q_mod_Bsk;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_base_matrix_Bsk_q(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Data64> base_matrix_Bsk_q;
        for (int k = 0; k < size; k++)
        {
            for (int i = 0; i < bsk_mod.size() - 1; i++)
            {
                Data64 temp = 1;
                for (int j = 0; j < bsk_mod.size() - 1; j++)
                {
                    if (i != j)
                    {
                        Data64 base = bsk_mod[j].value % primes[k].value;
                        temp = OPERATOR64::mult(temp, base, primes[k]);
                    }
                }
                base_matrix_Bsk_q.push_back(temp);
            }
        }

        return base_matrix_Bsk_q;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_base_change_matrix_msk(
        std::vector<Modulus64> bsk_mod)
    {
        std::vector<Data64> base_change_matrix_msk;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                if (i != j)
                {
                    temp = OPERATOR64::mult(temp, bsk_mod[j].value,
                                            bsk_mod[bsk_mod.size() - 1]);
                }
            }
            base_change_matrix_msk.push_back(temp);
        }

        return base_change_matrix_msk;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_inv_punctured_prod_mod_B_array(
        std::vector<Modulus64> bsk_mod)
    {
        std::vector<Data64> inv_punctured_prod_mod_B_array;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                if (i != j)
                {
                    temp = OPERATOR64::mult(temp, bsk_mod[j].value, bsk_mod[i]);
                }
            }
            inv_punctured_prod_mod_B_array.push_back(
                OPERATOR64::modinv(temp, bsk_mod[i]));
        }

        return inv_punctured_prod_mod_B_array;
    }

    Data64 HEContextImpl<Scheme::BFV>::generate_inv_prod_B_mod_m_sk(
        std::vector<Modulus64> bsk_mod)
    {
        Data64 inv_prod_B_mod_m_sk = 1;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            inv_prod_B_mod_m_sk =
                OPERATOR64::mult(inv_prod_B_mod_m_sk, bsk_mod[i].value,
                                 bsk_mod[bsk_mod.size() - 1]);
        }

        inv_prod_B_mod_m_sk = OPERATOR64::modinv(inv_prod_B_mod_m_sk,
                                                 bsk_mod[bsk_mod.size() - 1]);

        return inv_prod_B_mod_m_sk;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_prod_B_mod_q(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Data64> prod_B_mod_q;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                Data64 base = bsk_mod[j].value % primes[i].value;
                temp = OPERATOR64::mult(temp, base, primes[i]);
            }
            prod_B_mod_q.push_back(temp);
        }

        return prod_B_mod_q;
    }

    std::vector<Modulus64>
    HEContextImpl<Scheme::BFV>::generate_q_Bsk_merge_modulus(
        std::vector<Modulus64> primes, std::vector<Modulus64> bsk_mod, int size)
    {
        std::vector<Modulus64> q_Bsk_merge_modulus;
        for (int i = 0; i < size; i++)
        {
            q_Bsk_merge_modulus.push_back(primes[i]);
        }
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            q_Bsk_merge_modulus.push_back(bsk_mod[i]);
        }

        return q_Bsk_merge_modulus;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_q_Bsk_merge_root(
        std::vector<Data64> primes_psi, std::vector<Data64> bsk_mod_psi,
        int size)
    {
        std::vector<Data64> q_Bsk_merge_psi;
        for (int i = 0; i < size; i++)
        {
            q_Bsk_merge_psi.push_back(primes_psi[i]);
        }
        for (int i = 0; i < bsk_mod_psi.size(); i++)
        {
            q_Bsk_merge_psi.push_back(bsk_mod_psi[i]);
        }

        return q_Bsk_merge_psi;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_Qi_t(std::vector<Modulus64> primes,
                                              Modulus64& plain_mod, int size)
    {
        std::vector<Data64> Qi_t;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data64 mod_ = primes[j].value % plain_mod.value;
                    temp = OPERATOR64::mult(temp, mod_, plain_mod);
                }
            }
            Qi_t.push_back(temp);
        }

        return Qi_t;
    }

    std::vector<Data64>
    HEContextImpl<Scheme::BFV>::generate_Qi_gamma(std::vector<Modulus64> primes,
                                                  Modulus64& gamma, int size)
    {
        std::vector<Data64> Qi_gamma;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data64 mod = primes[j].value % gamma.value;
                    temp = OPERATOR64::mult(temp, mod, gamma);
                }
            }
            Qi_gamma.push_back(temp);
        }

        return Qi_gamma;
    }

    std::vector<Data64> HEContextImpl<Scheme::BFV>::generate_Qi_inverse(
        std::vector<Modulus64> primes, int size)
    {
        std::vector<Data64> Qi_inverse;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data64 mod_ = primes[j].value % primes[i].value;
                    Data64 inv_ = OPERATOR64::modinv(mod_, primes[i]);
                    temp = OPERATOR64::mult(temp, inv_, primes[i]);
                }
            }
            Qi_inverse.push_back(temp);
        }

        return Qi_inverse;
    }

    Data64 HEContextImpl<Scheme::BFV>::generate_mulq_inv_t(
        std::vector<Modulus64> primes, Modulus64& plain_mod, int size)
    {
        Data64 mulq_inv_t = 1;
        for (int i = 0; i < size; i++)
        {
            Data64 mod_ = primes[i].value % plain_mod.value;
            Data64 inv_ = OPERATOR64::modinv(mod_, plain_mod);
            mulq_inv_t = OPERATOR64::mult(mulq_inv_t, inv_, plain_mod);
        }

        mulq_inv_t = plain_mod.value - mulq_inv_t;

        return mulq_inv_t;
    }

    Data64 HEContextImpl<Scheme::BFV>::generate_mulq_inv_gamma(
        std::vector<Modulus64> primes, Modulus64& gamma, int size)
    {
        Data64 mulq_inv_gamma = 1;
        for (int i = 0; i < size; i++)
        {
            Data64 mod_ = primes[i].value % gamma.value;
            Data64 inv_ = OPERATOR64::modinv(mod_, gamma);
            mulq_inv_gamma = OPERATOR64::mult(mulq_inv_gamma, inv_, gamma);
        }

        mulq_inv_gamma = gamma.value - mulq_inv_gamma;

        return mulq_inv_gamma;
    }

    Data64 HEContextImpl<Scheme::BFV>::generate_inv_gamma(Modulus64& plain_mod,
                                                          Modulus64& gamma)
    {
        Data64 mod_ = gamma.value % plain_mod.value;
        Data64 inv_gamma = OPERATOR64::modinv(mod_, plain_mod);

        return inv_gamma;
    }

    std::string HEContextImpl<Scheme::BFV>::to_string() const
    {
        if (!context_generated_)
        {
            return "HEContext<BFV>(not generated)";
        }
        std::ostringstream stream;
        auto bits = [&](const std::vector<int>& sizes)
        {
            for (std::size_t i = 0; i < sizes.size(); i++)
            {
                stream << (i == 0 ? "" : "+") << sizes[i];
            }
        };
        stream << "HEContext<BFV>(N=" << n << ", Q=";
        bits(Q_mod_bit_sizes_);
        stream << " bits, P=";
        bits(P_mod_bit_sizes_);
        stream << " bits, t=" << plain_modulus_.value << ", "
               << detail::name(keyswitching_type_)
               << ", security=" << detail::name(sec_level_) << ", id=0x"
               << std::hex << context_id_ << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEContextImpl<Scheme::BFV>& object)
    {
        return os << object.to_string();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEContext<Scheme::BFV>& context)
    {
        if (!context)
        {
            return os << "HEContext<BFV>(null)";
        }
        return os << *context;
    }

    template class HEContextImpl<Scheme::BFV>;

} // namespace heongpu
