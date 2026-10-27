// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/context.cuh>
#include <heongpu/util/runtime.cuh>
#include <cmath>

namespace heongpu
{
    HEContextImpl<Scheme::CKKS>::HEContextImpl(const sec_level_type sec_level)
    {
        if (!coeff_modulus_specified_)
        {
            scheme_ = scheme_type::ckks;
            sec_level_ = sec_level;
            keyswitching_type_ = keyswitching_type::NONE;
        }
        else
        {
            throw std::logic_error("Parameters cannot be changed after the "
                                   "coeff_modulus is specified!");
        }
    }

    void HEContextImpl<Scheme::CKKS>::set_poly_modulus_degree(
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

    void HEContextImpl<Scheme::CKKS>::set_coeff_modulus_bit_sizes(
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

            // Digits of P_size primes; with one P prime they are single
            // primes, which earlier versions called KEYSWITCHING_METHOD_I.
            keyswitching_type_ = keyswitching_type::KEYSWITCHING_METHOD_II;

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

    void HEContextImpl<Scheme::CKKS>::set_coeff_modulus_values(
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

            // Digits of P_size primes; with one P prime they are single
            // primes, which earlier versions called KEYSWITCHING_METHOD_I.
            keyswitching_type_ = keyswitching_type::KEYSWITCHING_METHOD_II;

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
            throw std::logic_error("coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContextImpl<Scheme::CKKS>::set_rescale_mode(rescale_mode mode,
                                                       double scale)
    {
        if (!coeff_modulus_specified_)
        {
            throw StateError("The rescale mode needs the moduli! Set them "
                             "before set_rescale_mode.");
        }
        switch (mode)
        {
            case rescale_mode::manual:
                level_scales_.clear();
                break;
            case rescale_mode::automatic:
            {
                if (!(scale > 0.0) || !std::isfinite(scale))
                {
                    throw ParameterError(error_message(
                        "Invalid scale ", scale,
                        " for the automatic rescale mode! Expected a "
                        "positive scale, the scale of the top level."));
                }
                // The scale a multiplication of two ciphertexts of level l
                // and a rescaling give, computed as the operators do.
                level_scales_.assign(Q_size, 0.0);
                level_scales_[Q_size - 1] = scale;
                for (int level = Q_size - 1; level > 0; level--)
                {
                    double square = level_scales_[level] * level_scales_[level];
                    level_scales_[level - 1] =
                        square /
                        static_cast<double>(prime_vector_[level].value);
                }
                break;
            }
            default:
                throw ParameterError(error_message(
                    "Invalid rescale mode ", static_cast<int>(mode), "!"));
        }
        rescale_mode_ = mode;
    }

    double HEContextImpl<Scheme::CKKS>::get_level_scale(int level) const
    {
        if (rescale_mode_ != rescale_mode::automatic)
        {
            throw StateError("Levels have scales only in the automatic "
                             "rescale mode!");
        }
        if ((level < 0) || (level >= static_cast<int>(level_scales_.size())))
        {
            throw ParameterError(error_message("Invalid level ", level,
                                               "! Expected 0 to ",
                                               level_scales_.size() - 1, "."));
        }
        return level_scales_[level];
    }

    void HEContextImpl<Scheme::CKKS>::generate()
    {
        generate(MemoryPoolConfig::Defaults());
    }

    void
    HEContextImpl<Scheme::CKKS>::generate(const MemoryPoolConfig& pool_config)
    {
        if ((!context_generated_) && (poly_modulus_degree_specified_) &&
            (coeff_modulus_specified_))
        {
            // Memory pool, random number generator and kernel stack size.
            heongpu::initialize(pool_config);

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

            // For Rescale parameters for all depth
            std::vector<Data64> rescale_last_q_modinv;
            std::vector<Data64> rescaled_half_mod;
            std::vector<Data64> rescaled_half;
            for (int j = 0; j < (Q_size - 1); j++)
            {
                int inner = (Q_size - 1) - j;
                rescaled_half.push_back(prime_vector_[inner].value >> 1);
                for (int i = 0; i < inner; i++)
                {
                    Data64 temp_ =
                        prime_vector_[inner].value % prime_vector_[i].value;
                    rescale_last_q_modinv.push_back(
                        OPERATOR64::modinv(temp_, prime_vector_[i]));
                    rescaled_half_mod.push_back(rescaled_half[j] %
                                                prime_vector_[i].value);
                }
            }

            rescaled_last_q_modinv_ =
                std::make_shared<DeviceVector<Data64>>(rescale_last_q_modinv);

            rescaled_half_mod_ =
                std::make_shared<DeviceVector<Data64>>(rescaled_half_mod);

            rescaled_half_ =
                std::make_shared<DeviceVector<Data64>>(rescaled_half);

            std::vector<Data64> Mi;
            std::vector<Data64> Mi_inv;
            std::vector<Data64> upper_half_threshold;
            std::vector<Data64> decryption_modulus;

            for (int i = 0; i < Q_size; i++)
            {
                int depth_Q_size = Q_size - i;

                // Mi
                std::vector<Data64> Mi_inner =
                    calculate_Mi(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size * depth_Q_size; j++)
                {
                    Mi.push_back(Mi_inner[j]);
                }

                // Mi_inv
                std::vector<Data64> Mi_inv_inner =
                    calculate_Mi_inv(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    Mi_inv.push_back(Mi_inv_inner[j]);
                }

                // upper_half_threshold
                std::vector<Data64> upper_half_threshold_inner =
                    calculate_upper_half_threshold(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    upper_half_threshold.push_back(
                        upper_half_threshold_inner[j]);
                }

                // decryption_modulus
                std::vector<Data64> M_inner =
                    calculate_M(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    decryption_modulus.push_back(M_inner[j]);
                }
            }

            Mi_ = std::make_shared<DeviceVector<Data64>>(Mi);

            Mi_inv_ = std::make_shared<DeviceVector<Data64>>(Mi_inv);

            upper_half_threshold_ =
                std::make_shared<DeviceVector<Data64>>(upper_half_threshold);

            decryption_modulus_ =
                std::make_shared<DeviceVector<Data64>>(decryption_modulus);

            // prime_location_leveled
            std::vector<int> prime_loc;
            int counter = Q_size;
            for (int i = 0; i < Q_size - 1; i++)
            {
                for (int j = 0; j < counter; j++)
                {
                    prime_loc.push_back(j);
                }
                counter--;
                for (int j = 0; j < P_size; j++)
                {
                    prime_loc.push_back(Q_size + j);
                }
            }

            prime_location_leveled =
                std::make_shared<DeviceVector<int>>(prime_loc);

            //////////////////////////////////

            switch (static_cast<int>(keyswitching_type_))
            {
                case 2: // KEYSWITCHING_METHOD_II
                {
                    KeySwitchParameterGenerator pool_ckks(
                        n, base_q, P_size, scheme_, keyswitching_type_);

                    m_leveled = pool_ckks.m;
                    l_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_Q_);
                    l_tilda_leveled = std::make_shared<std::vector<int>>(
                        pool_ckks.level_Qtilda_);

                    d_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_d_);

                    std::vector<std::vector<Data64>>
                        base_change_matrix_D_to_Qtilda_vec =
                            pool_ckks.level_base_change_matrix_D_to_Qtilda();

                    std::vector<std::vector<Data64>> Mi_inv_D_to_Qtilda_vec =
                        pool_ckks.level_Mi_inv_D_to_Qtilda();

                    std::vector<std::vector<Data64>> prod_D_to_Qtilda_vec =
                        pool_ckks.level_prod_D_to_Qtilda();

                    std::vector<std::vector<int>> I_j_vec =
                        pool_ckks.level_I_j();

                    std::vector<std::vector<int>> I_location_vec =
                        pool_ckks.level_I_location();

                    std::vector<std::vector<int>> Sk_pair_new_vec =
                        pool_ckks.level_sk_pair();

                    base_change_matrix_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    Mi_inv_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    prod_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    I_j_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    I_location_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    Sk_pair_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();

                    for (int pool_lp = 0;
                         pool_lp < base_change_matrix_D_to_Qtilda_vec.size();
                         pool_lp++)
                    {
                        DeviceVector<Data64>
                            base_change_matrix_D_to_Qtilda_leveled_inner(
                                base_change_matrix_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data64> Mi_inv_D_to_Qtilda_leveled_inner(
                            Mi_inv_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data64> prod_D_to_Qtilda_leveled_inner(
                            prod_D_to_Qtilda_vec[pool_lp]);

                        base_change_matrix_D_to_Qtilda_leveled->push_back(
                            std::move(
                                base_change_matrix_D_to_Qtilda_leveled_inner));
                        Mi_inv_D_to_Qtilda_leveled->push_back(
                            std::move(Mi_inv_D_to_Qtilda_leveled_inner));
                        prod_D_to_Qtilda_leveled->push_back(
                            std::move(prod_D_to_Qtilda_leveled_inner));

                        DeviceVector<int> I_j_vec_inner(I_j_vec[pool_lp]);
                        DeviceVector<int> I_location_vec_inner(
                            I_location_vec[pool_lp]);
                        DeviceVector<int> Sk_pair_new_vec_inner(
                            Sk_pair_new_vec[pool_lp]);

                        I_j_leveled->push_back(std::move(I_j_vec_inner));
                        I_location_leveled->push_back(
                            std::move(I_location_vec_inner));
                        Sk_pair_leveled->push_back(
                            std::move(Sk_pair_new_vec_inner));
                    }
                }
                break;
                default:
                    throw ParameterError(error_message(
                        "Invalid Key Switching Type ",
                        static_cast<int>(keyswitching_type_),
                        "! Expected KEYSWITCHING_METHOD_II (2)."));
                    break;
            }
            context_id_ = context_identity(static_cast<int>(scheme_), n,
                                           prime_vector_, 0);
            context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Context is already generated!");
        }
    }

    void HEContextImpl<Scheme::CKKS>::print_parameters(std::ostream& os) const
    {
        if (context_generated_)
        {
            std::string scheme_string = "CKKS";

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

            os << std::endl;
        }
        else
        {
            os << "Parameters is not generated yet!" << std::endl;
        }
    }

    void HEContextImpl<Scheme::CKKS>::save(std::ostream& os,
                                           serialization_format format) const
    {
        if ((poly_modulus_degree_specified_) && (coeff_modulus_specified_))
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
        }
        else
        {
            throw std::runtime_error(
                "Context has no enough parameters to serialize!");
        }
    }

    void HEContextImpl<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!context_generated_))
        {
            const std::uint64_t loaded_context_id =
                detail::read_serialization_header(
                    is, detail::serialized_object::context, scheme_type::ckks);
            detail::validate_loaded_context_identity(
                context_id_, loaded_context_id, "HEContext");

            is.read((char*) &scheme_, sizeof(scheme_));

            is.read((char*) &sec_level_, sizeof(sec_level_));

            is.read((char*) &keyswitching_type_, sizeof(keyswitching_type_));
            // KEYSWITCHING_METHOD_I of earlier versions is method II with
            // single prime digits.
            keyswitching_type_ = keyswitching_type::KEYSWITCHING_METHOD_II;

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

            poly_modulus_degree_specified_ = true;
            coeff_modulus_specified_ = true;
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

    std::string HEContextImpl<Scheme::CKKS>::to_string() const
    {
        if (!context_generated_)
        {
            return "HEContext<CKKS>(not generated)";
        }
        std::ostringstream stream;
        auto bits = [&](const std::vector<int>& sizes)
        {
            for (std::size_t i = 0; i < sizes.size(); i++)
            {
                stream << (i == 0 ? "" : "+") << sizes[i];
            }
        };
        stream << "HEContext<CKKS>(N=" << n << ", Q=";
        bits(Q_mod_bit_sizes_);
        stream << " bits, P=";
        bits(P_mod_bit_sizes_);
        stream << " bits, " << detail::name(keyswitching_type_)
               << ", security=" << detail::name(sec_level_);
        if (rescale_mode_ != rescale_mode::manual)
        {
            stream << ", rescale=" << detail::name(rescale_mode_);
        }
        stream << ", id=0x" << std::hex << context_id_ << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEContextImpl<Scheme::CKKS>& object)
    {
        return os << object.to_string();
    }

    std::ostream& operator<<(std::ostream& os,
                             const HEContext<Scheme::CKKS>& context)
    {
        if (!context)
        {
            return os << "HEContext<CKKS>(null)";
        }
        return os << *context;
    }

    template class HEContextImpl<Scheme::CKKS>;

} // namespace heongpu
