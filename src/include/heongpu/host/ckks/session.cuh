// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_SESSION_H
#define HEONGPU_CKKS_SESSION_H

#include <functional>
#include <memory>
#include <vector>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/secretkey.cuh>
#include <heongpu/host/ckks/publickey.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/encryptor.cuh>
#include <heongpu/host/ckks/decryptor.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/parameters.cuh>

namespace heongpu
{
    namespace detail
    {
        template <Scheme S> struct SessionState;
    } // namespace detail

    /**
     * @brief A ciphertext of an HESession. The operators +, -, * and their
     * compound assignments work on ciphertexts of the same session, on
     * constants and on vectors of values, with the keys of the session.
     *
     * A product of two ciphertexts is relinearized. In
     * rescale_mode::automatic, the default of HESession, the operands are
     * rescaled and brought to the same level when an operation needs it; in
     * rescale_mode::manual, the application rescales with the operators of
     * the session, see HESession::operators().
     *
     * The ciphertext keeps the state of its session: it stays usable after
     * the HESession object it came from is destroyed.
     */
    template <> class SessionCiphertext<Scheme::CKKS>
    {
        friend class HESession<Scheme::CKKS>;

      public:
        // An empty ciphertext, which can only be assigned to.
        SessionCiphertext() = default;

        // The ciphertext of the lower level API.
        Ciphertext<Scheme::CKKS>& ciphertext() noexcept { return ciphertext_; }
        const Ciphertext<Scheme::CKKS>& ciphertext() const noexcept
        {
            return ciphertext_;
        }

        int level() const noexcept { return ciphertext_.level(); }
        int depth() const noexcept { return ciphertext_.depth(); }
        double scale() const noexcept { return ciphertext_.scale(); }

        /**
         * @brief Returns the ciphertext with its slots rotated left by shift
         * (right for a negative shift). The session generates the Galois key
         * for the shift the first time it is used, unless it was given to
         * the session up front.
         */
        SessionCiphertext rotate(int shift) const;

        /**
         * @brief Returns the complex conjugate of the slots.
         */
        SessionCiphertext conjugate() const;

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
        SessionCiphertext& operator+=(double value)
        {
            return *this = add(value);
        }
        SessionCiphertext& operator-=(double value)
        {
            return *this = add(-value);
        }
        SessionCiphertext& operator*=(double value)
        {
            return *this = multiply(value);
        }
        // The values are encoded one per slot; the other slots are 0.
        SessionCiphertext& operator+=(const std::vector<double>& values)
        {
            return *this = add(values);
        }
        SessionCiphertext& operator-=(const std::vector<double>& values)
        {
            return *this = sub(values);
        }
        SessionCiphertext& operator*=(const std::vector<double>& values)
        {
            return *this = multiply(values);
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
                                           double input2)
        {
            return input1.add(input2);
        }
        friend SessionCiphertext operator+(double input1,
                                           const SessionCiphertext& input2)
        {
            return input2.add(input1);
        }
        friend SessionCiphertext operator-(const SessionCiphertext& input1,
                                           double input2)
        {
            return input1.add(-input2);
        }
        friend SessionCiphertext operator-(double input1,
                                           const SessionCiphertext& input2)
        {
            return input2.negate().add(input1);
        }
        friend SessionCiphertext operator*(const SessionCiphertext& input1,
                                           double input2)
        {
            return input1.multiply(input2);
        }
        friend SessionCiphertext operator*(double input1,
                                           const SessionCiphertext& input2)
        {
            return input2.multiply(input1);
        }

        friend SessionCiphertext operator+(const SessionCiphertext& input1,
                                           const std::vector<double>& input2)
        {
            return input1.add(input2);
        }
        friend SessionCiphertext operator+(const std::vector<double>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.add(input1);
        }
        friend SessionCiphertext operator-(const SessionCiphertext& input1,
                                           const std::vector<double>& input2)
        {
            return input1.sub(input2);
        }
        friend SessionCiphertext operator-(const std::vector<double>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.negate().add(input1);
        }
        friend SessionCiphertext operator*(const SessionCiphertext& input1,
                                           const std::vector<double>& input2)
        {
            return input1.multiply(input2);
        }
        friend SessionCiphertext operator*(const std::vector<double>& input1,
                                           const SessionCiphertext& input2)
        {
            return input2.multiply(input1);
        }

      private:
        SessionCiphertext(
            std::shared_ptr<detail::SessionState<Scheme::CKKS>> state,
            Ciphertext<Scheme::CKKS> ciphertext);

        SessionCiphertext add(const SessionCiphertext& other) const;
        SessionCiphertext sub(const SessionCiphertext& other) const;
        // Relinearizes the product.
        SessionCiphertext multiply(const SessionCiphertext& other) const;
        SessionCiphertext negate() const;
        SessionCiphertext add(double value) const;
        SessionCiphertext multiply(double value) const;
        SessionCiphertext add(const std::vector<double>& values) const;
        SessionCiphertext sub(const std::vector<double>& values) const;
        SessionCiphertext multiply(const std::vector<double>& values) const;

        // Throws StateError for an empty ciphertext.
        detail::SessionState<Scheme::CKKS>& state() const;

        // Throws ContextMismatchError unless other is of the same session.
        void check_session(const SessionCiphertext& other) const;

        // values encoded as the operand of an addition or a multiplication
        // with this ciphertext: at its level, and with its scale or the
        // scale a product needs.
        Plaintext<Scheme::CKKS> encode(const std::vector<double>& values,
                                       bool multiply) const;

        SessionCiphertext result(Ciphertext<Scheme::CKKS> ciphertext) const;

        std::shared_ptr<detail::SessionState<Scheme::CKKS>> state_;
        // Mutable because the operators take their inputs by non-const
        // reference without changing them.
        mutable Ciphertext<Scheme::CKKS> ciphertext_;
    };

    /**
     * @brief One party's CKKS computation in one object: the context, the
     * keys, the encoder, the encryptor, the decryptor and the operators.
     *
     * @code
     * heongpu::HEParams<heongpu::Scheme::CKKS> params;
     * params.multiplicative_depth = 3;
     * heongpu::HESession<heongpu::Scheme::CKKS> session(params);
     * auto x = session.encrypt(xs);
     * auto y = session.encrypt(ys);
     * auto z = x * x * y + 0.5 * x - y;
     * std::vector<double> result = session.decrypt(z);
     * @endcode
     *
     * The session holds the secret key: it is for the party that encrypts
     * and decrypts, not for a server that computes on ciphertexts it
     * receives. Copies of the session share its state.
     */
    template <> class HESession<Scheme::CKKS>
    {
      public:
        /**
         * @brief Generates the context of params.generate() and the secret,
         * public and relinearization keys, and the Galois keys for the shifts
         * in rotations.
         *
         * The rescale mode is params.rescale_mode, and
         * rescale_mode::automatic if it is not set.
         */
        explicit HESession(const HEParams<Scheme::CKKS>& params,
                           const std::vector<int>& rotations = {});

        /**
         * @brief Generates the context of parameters, with its rescale mode,
         * and the keys as HESession(params, rotations) does.
         */
        explicit HESession(const HEParameterSet<Scheme::CKKS>& parameters,
                           const std::vector<int>& rotations = {});

        /**
         * @brief Returns the values encrypted at the top level with the scale
         * of the parameters, one value per slot; the other slots are 0.
         */
        SessionCiphertext<Scheme::CKKS>
        encrypt(const std::vector<double>& values) const;
        SessionCiphertext<Scheme::CKKS>
        encrypt(const std::vector<Complex64>& values) const;

        /**
         * @brief Returns plain encrypted.
         */
        SessionCiphertext<Scheme::CKKS>
        encrypt(Plaintext<Scheme::CKKS>& plain) const;

        /**
         * @brief Returns ciphertext as a ciphertext of the session. Throws
         * ContextMismatchError unless it has the parameters of the session;
         * it must also be encrypted with the keys of the session, which the
         * ciphertext does not record.
         */
        SessionCiphertext<Scheme::CKKS>
        wrap(Ciphertext<Scheme::CKKS> ciphertext) const;

        /**
         * @brief Returns the values of all slots of input.
         */
        std::vector<double>
        decrypt(const SessionCiphertext<Scheme::CKKS>& input) const;
        std::vector<Complex64>
        decrypt_complex(const SessionCiphertext<Scheme::CKKS>& input) const;

        /**
         * @brief See HEArithmeticOperator<Scheme::CKKS>::evaluate_polynomial.
         */
        SessionCiphertext<Scheme::CKKS>
        evaluate_polynomial(const SessionCiphertext<Scheme::CKKS>& input,
                            const std::vector<double>& coefficients) const;

        /**
         * @brief See HEArithmeticOperator<Scheme::CKKS>::evaluate_chebyshev.
         */
        SessionCiphertext<Scheme::CKKS>
        evaluate_chebyshev(const SessionCiphertext<Scheme::CKKS>& input,
                           const std::vector<double>& coefficients, double a,
                           double b) const;
        SessionCiphertext<Scheme::CKKS>
        evaluate_chebyshev(const SessionCiphertext<Scheme::CKKS>& input,
                           const std::function<double(double)>& function,
                           double a, double b, int degree) const;

        /**
         * @brief Generates the Galois keys for the shifts the session does
         * not have a key for yet.
         */
        void add_rotations(const std::vector<int>& shifts) const;

        int slot_count() const;
        // The scale of the parameters, which encrypt() uses.
        double scale() const;
        rescale_mode get_rescale_mode() const;

        // The objects of the lower level API.
        HEContext<Scheme::CKKS> context() const;
        Secretkey<Scheme::CKKS>& secret_key() const;
        Publickey<Scheme::CKKS>& public_key() const;
        Relinkey<Scheme::CKKS>& relin_key() const;
        // The Galois keys generated so far, with the conjugation key.
        Galoiskey<Scheme::CKKS>& galois_key() const;
        HEEncoder<Scheme::CKKS>& encoder() const;
        HEEncryptor<Scheme::CKKS>& encryptor() const;
        HEDecryptor<Scheme::CKKS>& decryptor() const;
        HEArithmeticOperator<Scheme::CKKS>& operators() const;

      private:
        friend class SessionCiphertext<Scheme::CKKS>;

        SessionCiphertext<Scheme::CKKS>
        result(Ciphertext<Scheme::CKKS> ciphertext) const;

        // Throws ContextMismatchError unless input is of this session.
        void check_session(const SessionCiphertext<Scheme::CKKS>& input) const;

        // Generates the Galois keys of state for the shifts it has no key
        // for, and the key with the conjugation key alone if it has no keys.
        static void
        generate_rotations(detail::SessionState<Scheme::CKKS>& state,
                           const std::vector<int>& shifts);

        std::shared_ptr<detail::SessionState<Scheme::CKKS>> state_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_SESSION_H
