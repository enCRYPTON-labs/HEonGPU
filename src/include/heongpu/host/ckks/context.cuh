// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_CONTEXT_H
#define HEONGPU_CKKS_CONTEXT_H

#include <heongpu/util/util.cuh>
#include <heongpu/util/schemes.h>
#include <heongpu/util/devicevector.cuh>
#include <heongpu/util/hostvector.cuh>
#include <heongpu/util/secstdparams.h>
#include <heongpu/util/defaultmodulus.hpp>
#include <heongpu/util/random.cuh>
#include <heongpu/util/memorypool.cuh>
#include <gmp.h>
#include <heongpu/kernel/contextpool.hpp>

#include <ostream>
#include <istream>

namespace heongpu
{
    template <> class HEContextImpl<Scheme::CKKS>
    {
        template <Scheme S> friend class Secretkey;
        template <Scheme S> friend class Publickey;
        template <Scheme S> friend class MultipartyPublickey;
        template <Scheme S> friend class Plaintext;
        template <Scheme S> friend class Ciphertext;
        template <Scheme S> friend class Relinkey;
        template <Scheme S> friend class MultipartyRelinkey;
        template <Scheme S> friend class Galoiskey;
        template <Scheme S> friend class MultipartyGaloiskey;
        template <Scheme S> friend class Switchkey;
        template <Scheme S> friend class HEEncoder;
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;

      public:
        HEContextImpl(const sec_level_type = sec_level_type::sec128);

        void set_poly_modulus_degree(size_t poly_modulus_degree);

        void set_coeff_modulus_bit_sizes(
            const std::vector<int>& log_Q_bases_bit_sizes,
            const std::vector<int>& log_P_bases_bit_sizes);

        void set_coeff_modulus_values(const std::vector<Data64>& log_Q_bases,
                                      const std::vector<Data64>& log_P_bases);

        void generate();
        void generate(const MemoryPoolConfig& pool_config);

        /**
         * @brief Writes the encryption parameters to os, std::cout by
         * default.
         */
        void print_parameters(std::ostream& os = std::cout) const;

        /**
         * @brief Returns a one line description, for example for logs and
         * error reports. Also written by operator<<.
         */
        std::string to_string() const;

        inline int get_poly_modulus_degree() const noexcept { return n; }

        inline int get_log_poly_modulus_degree() const noexcept
        {
            return n_power;
        }

        inline int get_ciphertext_modulus_count() const noexcept
        {
            return Q_size;
        }

        inline int get_key_modulus_count() const noexcept
        {
            return Q_prime_size;
        }

        inline std::vector<Modulus64> get_key_modulus() const noexcept
        {
            return prime_vector_;
        }

        /**
         * @brief Returns the identity of the context, computed in generate()
         * from the parameters the objects of the context depend on. Contexts
         * generated from the same parameters have the same identity, and
         * objects can only be used with a context that has the identity of
         * the context they were created with. Returns 0 before generate().
         */
        inline std::uint64_t context_id() const noexcept { return context_id_; }

        void save(std::ostream& os,
                  serialization_format format = serialization_format::v2) const;

        void load(std::istream& is);

      private:
        bool poly_modulus_degree_specified_ = false;
        bool coeff_modulus_specified_ = false;
        bool context_generated_ = false;
        std::uint64_t context_id_ = 0;

        scheme_type scheme_;
        sec_level_type sec_level_;
        keyswitching_type keyswitching_type_;

        int n;
        int n_power;

        int coeff_modulus;
        int total_coeff_bit_count;

        int Q_prime_size;
        int Q_size;
        int P_size;

        std::vector<Modulus64> prime_vector_;
        std::vector<Data64> base_q;

        std::vector<int> Qprime_mod_bit_sizes_;
        std::vector<int> Q_mod_bit_sizes_;
        std::vector<int> P_mod_bit_sizes_;

        /////////////

        int total_bit_count_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;
        std::shared_ptr<DeviceVector<Data64>> half_p_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;
        std::shared_ptr<DeviceVector<Data64>> factor_;

        // CKKS switchkey & rescale parameters
        std::shared_ptr<DeviceVector<Data64>> rescaled_last_q_modinv_;
        std::shared_ptr<DeviceVector<Data64>> rescaled_half_;
        std::shared_ptr<DeviceVector<Data64>> rescaled_half_mod_;

        // CKKS encode & decode parameters
        // parameters
        std::shared_ptr<DeviceVector<Data64>> Mi_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data64>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data64>> decryption_modulus_;

        // CKKS switchkey parameters (Method II)
        int m_leveled;
        std::shared_ptr<std::vector<int>> l_leveled;
        std::shared_ptr<std::vector<int>> l_tilda_leveled;
        std::shared_ptr<std::vector<int>> d_leveled;
        std::shared_ptr<std::vector<int>> d_tilda_leveled;
        int r_prime_leveled;

        std::shared_ptr<DeviceVector<Modulus64>> B_prime_leveled;
        std::shared_ptr<DeviceVector<Root64>> B_prime_ntt_tables_leveled;
        std::shared_ptr<DeviceVector<Root64>> B_prime_intt_tables_leveled;
        std::shared_ptr<DeviceVector<Ninverse64>> B_prime_n_inverse_leveled;

        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_D_to_B_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_B_to_D_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            Mi_inv_D_to_B_leveled;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_B_to_D_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>> prod_D_to_B_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>> prod_B_to_D_leveled;

        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_D_to_Qtilda_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            Mi_inv_D_to_Qtilda_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            prod_D_to_Qtilda_leveled;

        std::shared_ptr<std::vector<DeviceVector<int>>> I_j_leveled;
        std::shared_ptr<std::vector<DeviceVector<int>>> I_location_leveled;
        std::shared_ptr<std::vector<DeviceVector<int>>> Sk_pair_leveled;

        // int* prime_location_leveled;
        std::shared_ptr<DeviceVector<int>> prime_location_leveled;
    };

    std::ostream& operator<<(std::ostream& os,
                             const HEContextImpl<Scheme::CKKS>& object);

    std::ostream& operator<<(std::ostream& os,
                             const HEContext<Scheme::CKKS>& context);

} // namespace heongpu
#endif // HEONGPU_CKKS_CONTEXT_H
