// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_SESSION_H
#define HEONGPU_BFV_SESSION_H

#include <concepts>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <shared_mutex>
#include <vector>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/secretkey.cuh>
#include <heongpu/host/bfv/publickey.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/ciphertext.cuh>
#include <heongpu/host/bfv/evaluationkey.cuh>
#include <heongpu/host/bfv/encoder.cuh>
#include <heongpu/host/bfv/encryptor.cuh>
#include <heongpu/host/bfv/decryptor.cuh>
#include <heongpu/host/bfv/operator.cuh>
#include <heongpu/host/bfv/parameters.cuh>

namespace heongpu
{
    namespace detail
    {
        template <Scheme S> struct SessionState;

        // The slot values of BFV: integers modulo the plain modulus, given
        // as unsigned or signed values.
        template <typename T>
        concept bfv_slot_value =
            std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t>;
    } // namespace detail

    /**
     * @brief A ciphertext of an HESession. The operators +, -, * and their
     * compound assignments work on ciphertexts of the same session, on
     * integer constants and on vectors of slot values, with the keys of the
     * session. A product of two ciphertexts is relinearized.
     *
     * The ciphertext keeps the state of its session: it stays usable after
     * the HESession object it came from is destroyed.
     */
    template <> class SessionCiphertext<Scheme::BFV>
    {
        friend class HESession<Scheme::BFV>;

      public:
        // An empty ciphertext, which can only be assigned to.
        SessionCiphertext() = default;

        // The ciphertext of the lower level API.
        Ciphertext<Scheme::BFV>& ciphertext() noexcept { return ciphertext_; }
        const Ciphertext<Scheme::BFV>& ciphertext() const noexcept
        {
            return ciphertext_;
        }

        /**
         * @brief Returns the ciphertext with both rows of slots rotated left
         * by shift (right for a negative shift). The session generates the
         * Galois key for the shift the first time it is used, unless it was
         * given to the session up front.
         */
        SessionCiphertext rotate(int shift) const;

        /**
         * @brief Returns the ciphertext with the two rows of slots swapped.
         */
        SessionCiphertext rotate_columns() const;

        /**
         * @brief Returns the ciphertext rotated by each of shifts, as
         * rotate(shift) does; the rotations share the decomposition of the
         * ciphertext, see HEArithmeticOperator<Scheme::BFV>::
         * rotate_rows_hoisted.
         */
        std::vector<SessionCiphertext>
        rotate_hoisted(const std::vector<int>& shifts) const;

        /**
         * @brief Returns a ciphertext whose slots all hold the sum of the
         * slots of both rows. Uses the rotations by 1, 2, 4, ...,
         * row size / 2 and the column rotation.
         */
        SessionCiphertext sum_slots() const;

        /**
         * @brief Returns a ciphertext whose slots all hold the inner product
         * of the slots with those of other, or with values, one per slot:
         * the product and then sum_slots().
         */
        SessionCiphertext inner_product(const SessionCiphertext& other) const
        {
            return multiply(other).sum_slots();
        }
        template <detail::bfv_slot_value T>
        SessionCiphertext inner_product(const std::vector<T>& values) const
        {
            return multiply(encode(values)).sum_slots();
        }

        /**
         * @brief Writes the ciphertext, which HESession::load_ciphertext of
         * a session with the same keys reads.
         */
        void save(std::ostream& os) const { ciphertext_.save(os); }

        SessionCiphertext& operator+=(const SessionCiphertext& other)
        {
            return *this = add(other);
        }
        SessionCiphertext& operator-=(const SessionCiphertext& other)
        {
            return *this = sub(other);
        }
        SessionCiphertext& operator*=(const SessionCiphertext& other)
        {
            return *this = multiply(other);
        }
        // A constant is added to or multiplied with every slot.
        SessionCiphertext& operator+=(std::int64_t value)
        {
            return *this = add(constant(value));
        }
        SessionCiphertext& operator-=(std::int64_t value)
        {
            return *this = sub(constant(value));
        }
        SessionCiphertext& operator*=(std::int64_t value)
        {
            return *this = multiply(constant(value));
        }
        // The values are encoded one per slot; the other slots are 0.
        template <detail::bfv_slot_value T>
        SessionCiphertext& operator+=(const std::vector<T>& values)
        {
            return *this = add(encode(values));
        }
        template <detail::bfv_slot_value T>
        SessionCiphertext& operator-=(const std::vector<T>& values)
        {
            return *this = sub(encode(values));
        }
        template <detail::bfv_slot_value T>
        SessionCiphertext& operator*=(const std::vector<T>& values)
        {
            return *this = multiply(encode(values));
        }

        friend SessionCiphertext operator+(const SessionCiphertext& input1,
                                           const SessionCiphertext& input2)
        {
            return input1.add(input2);
        }
        friend SessionCiphertext operator-(const SessionCiphertext& input1,
                                           const SessionCiphertext& input2)
        {
            return input1.sub(input2);
        }
        friend SessionCiphertext operator*(const SessionCiphertext& input1,
                                           const SessionCiphertext& input2)
        {
            return input1.multiply(input2);
        }
        friend SessionCiphertext operator-(const SessionCiphertext& input)
        {
            return input.negate();
        }

        friend SessionCiphertext operator+(const SessionCiphertext& input1,
                                           std::int64_t input2)
        {
            return input1.add(input1.constant(input2));
        }
        friend SessionCiphertext operator+(std::int64_t input1,
                                           const SessionCiphertext& input2)
        {
            return input2.add(input2.constant(input1));
        }
        friend SessionCiphertext operator-(const SessionCiphertext& input1,
                                           std::int64_t input2)
        {
            return input1.sub(input1.constant(input2));
        }
        friend SessionCiphertext operator-(std::int64_t input1,
                                           const SessionCiphertext& input2)
        {
            return input2.negate().add(input2.constant(input1));
        }
        friend SessionCiphertext operator*(const SessionCiphertext& input1,
                                           std::int64_t input2)
        {
            return input1.multiply(input1.constant(input2));
        }
        friend SessionCiphertext operator*(std::int64_t input1,
                                           const SessionCiphertext& input2)
        {
            return input2.multiply(input2.constant(input1));
        }

        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator+(const SessionCiphertext& input1,
                                           const std::vector<T>& input2)
        {
            return input1.add(input1.encode(input2));
        }
        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator+(const std::vector<T>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.add(input2.encode(input1));
        }
        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator-(const SessionCiphertext& input1,
                                           const std::vector<T>& input2)
        {
            return input1.sub(input1.encode(input2));
        }
        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator-(const std::vector<T>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.negate().add(input2.encode(input1));
        }
        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator*(const SessionCiphertext& input1,
                                           const std::vector<T>& input2)
        {
            return input1.multiply(input1.encode(input2));
        }
        template <detail::bfv_slot_value T>
        friend SessionCiphertext operator*(const std::vector<T>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.multiply(input2.encode(input1));
        }

      private:
        SessionCiphertext(
            std::shared_ptr<detail::SessionState<Scheme::BFV>> state,
            Ciphertext<Scheme::BFV> ciphertext);

        SessionCiphertext add(const SessionCiphertext& other) const;
        SessionCiphertext sub(const SessionCiphertext& other) const;
        // Relinearizes the product.
        SessionCiphertext multiply(const SessionCiphertext& other) const;
        SessionCiphertext negate() const;
        SessionCiphertext add(Plaintext<Scheme::BFV> plain) const;
        SessionCiphertext sub(Plaintext<Scheme::BFV> plain) const;
        SessionCiphertext multiply(Plaintext<Scheme::BFV> plain) const;

        Plaintext<Scheme::BFV>
        encode(const std::vector<std::uint64_t>& values) const;
        Plaintext<Scheme::BFV>
        encode(const std::vector<std::int64_t>& values) const;
        // value in every slot.
        Plaintext<Scheme::BFV> constant(std::int64_t value) const;

        // Throws StateError for an empty ciphertext.
        detail::SessionState<Scheme::BFV>& state() const;

        // Throws ContextMismatchError unless other is of the same session.
        void check_session(const SessionCiphertext& other) const;

        SessionCiphertext result(Ciphertext<Scheme::BFV> ciphertext) const;

        std::shared_ptr<detail::SessionState<Scheme::BFV>> state_;
        Ciphertext<Scheme::BFV> ciphertext_;
    };

    /**
     * @brief One party's BFV computation in one object: the context, the
     * keys, the encoder, the encryptor, the decryptor and the operators.
     *
     * @code
     * heongpu::HEParams<heongpu::Scheme::BFV> params;
     * params.multiplicative_depth = 2;
     * heongpu::HESession<heongpu::Scheme::BFV> session(params);
     * auto x = session.encrypt(xs);
     * auto y = session.encrypt(ys);
     * auto z = x * x * y + 3 * x - y;
     * std::vector<std::uint64_t> result = session.decrypt(z);
     * @endcode
     *
     * A session made from parameters is the owner of its keys. save() writes
     * the session with the keys of a role, and load() reads it, for example
     * for a server that computes without the secret key; see
     * HESession<Scheme::CKKS>. Copies of the session share its state, and
     * threads can use them at the same time.
     */
    template <> class HESession<Scheme::BFV>
    {
      public:
        /**
         * @brief Generates the context of params.generate() and the secret,
         * public and relinearization keys, and the Galois keys for the shifts
         * in rotations.
         */
        explicit HESession(const HEParams<Scheme::BFV>& params,
                           const std::vector<int>& rotations = {});

        /**
         * @brief Generates the context of parameters and the keys as
         * HESession(params, rotations) does.
         */
        explicit HESession(const HEParameterSet<Scheme::BFV>& parameters,
                           const std::vector<int>& rotations = {});

        /**
         * @brief Returns the values encrypted, one per slot; the other slots
         * are 0. Signed values are taken modulo the plain modulus.
         */
        SessionCiphertext<Scheme::BFV>
        encrypt(const std::vector<std::uint64_t>& values) const;
        SessionCiphertext<Scheme::BFV>
        encrypt(const std::vector<std::int64_t>& values) const;

        /**
         * @brief Returns plain encrypted.
         */
        SessionCiphertext<Scheme::BFV>
        encrypt(const Plaintext<Scheme::BFV>& plain) const;

        /**
         * @brief Returns ciphertext as a ciphertext of the session. Throws
         * ContextMismatchError unless it has the parameters of the session;
         * it must also be encrypted with the keys of the session, which the
         * ciphertext does not record.
         */
        SessionCiphertext<Scheme::BFV>
        wrap(Ciphertext<Scheme::BFV> ciphertext) const;

        /**
         * @brief Returns the values of all slots of input, from 0 to the
         * plain modulus.
         */
        std::vector<std::uint64_t>
        decrypt(const SessionCiphertext<Scheme::BFV>& input) const;

        /**
         * @brief Returns the values of all slots of input, from -(t - 1) / 2
         * to (t - 1) / 2 for the plain modulus t.
         */
        std::vector<std::int64_t>
        decrypt_signed(const SessionCiphertext<Scheme::BFV>& input) const;

        /**
         * @brief Returns the bits of noise budget input has left; it decrypts
         * correctly while the budget is above 0.
         */
        int noise_budget(const SessionCiphertext<Scheme::BFV>& input) const;

        /**
         * @brief Generates the Galois keys for the shifts the session does
         * not have a key for yet.
         */
        void add_rotations(const std::vector<int>& shifts) const;

        /**
         * @brief Writes the session with the keys of role: the context, the
         * public key, for the evaluator also the relinearization and Galois
         * keys, and for the owner also the secret key. Throws StateError if
         * the session does not have them.
         */
        void save(std::ostream& os, session_role role) const;

        /**
         * @brief Reads a session that save() wrote. With expected, throws
         * StateError, before reading the keys, unless the session was saved
         * with that role.
         */
        static HESession load(std::istream& is);
        static HESession load(std::istream& is, session_role expected);

        /**
         * @brief Reads a ciphertext that SessionCiphertext::save or
         * Ciphertext::save wrote, which must have the parameters of the
         * session; see wrap().
         */
        SessionCiphertext<Scheme::BFV> load_ciphertext(std::istream& is) const;

        // The keys of the session: the owner has the secret key, the
        // evaluator the relinearization key, the encryptor neither.
        session_role role() const;

        int slot_count() const;
        Data64 plain_modulus() const;

        // The objects of the lower level API.
        HEContext<Scheme::BFV> context() const;
        Secretkey<Scheme::BFV>& secret_key() const;
        Publickey<Scheme::BFV>& public_key() const;
        Relinkey<Scheme::BFV>& relin_key() const;
        // The Galois keys generated so far, with the column rotation key. A
        // rotation by a new step adds its key to it: do not use it while other
        // threads rotate with the session.
        Galoiskey<Scheme::BFV>& galois_key() const;
        HEEncoder<Scheme::BFV>& encoder() const;
        HEEncryptor<Scheme::BFV>& encryptor() const;
        HEDecryptor<Scheme::BFV>& decryptor() const;
        HEArithmeticOperator<Scheme::BFV>& operators() const;

      private:
        friend class SessionCiphertext<Scheme::BFV>;

        explicit HESession(
            std::shared_ptr<detail::SessionState<Scheme::BFV>> state);

        // load(is) with expected, if it is not null.
        static HESession load_session(std::istream& is,
                                      const session_role* expected);

        SessionCiphertext<Scheme::BFV>
        result(Ciphertext<Scheme::BFV> ciphertext) const;

        // Throws ContextMismatchError unless input is of this session.
        void check_session(const SessionCiphertext<Scheme::BFV>& input) const;

        // Returns a shared lock of the Galois key of state, once it has the
        // keys of shifts and the column rotation key. The lock keeps other
        // threads from adding keys while the caller uses the key.
        static std::shared_lock<std::shared_mutex>
        rotation_keys(detail::SessionState<Scheme::BFV>& state,
                      const std::vector<int>& shifts);

        // Generates the Galois keys of state for the shifts it has no key
        // for, and the key with the column rotation key alone if it has no
        // keys. The caller holds the lock of the key exclusively.
        static void generate_rotations(detail::SessionState<Scheme::BFV>& state,
                                       const std::vector<int>& shifts);

        std::shared_ptr<detail::SessionState<Scheme::BFV>> state_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_SESSION_H
