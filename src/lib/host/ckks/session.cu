// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/session.cuh>
#include <heongpu/host/ckks/keygenerator.cuh>
#include <algorithm>
#include <mutex>
#include <optional>

namespace heongpu
{
    namespace detail
    {
        template <> struct SessionState<Scheme::CKKS>
        {
            explicit SessionState(
                const HEParameterSet<Scheme::CKKS>& parameters)
                : context(generated_context(parameters)),
                  scale(parameters.scale), keygen(context),
                  secret_key(keygen.secret_key()),
                  public_key(keygen.public_key(secret_key)),
                  relin_key(keygen.relin_key(secret_key)), encoder(context),
                  encryptor(context, public_key),
                  decryptor(context, secret_key), operators(context, encoder)
            {
            }

            static HEContext<Scheme::CKKS>
            generated_context(const HEParameterSet<Scheme::CKKS>& parameters)
            {
                HEContext<Scheme::CKKS> context = parameters.make_context();
                context->generate();
                return context;
            }

            HEContext<Scheme::CKKS> context;
            double scale;
            HEKeyGenerator<Scheme::CKKS> keygen;
            Secretkey<Scheme::CKKS> secret_key;
            Publickey<Scheme::CKKS> public_key;
            Relinkey<Scheme::CKKS> relin_key;
            // Created by the first rotation or conjugation, unless the
            // session is given rotations.
            std::optional<Galoiskey<Scheme::CKKS>> galois_key;
            // Held shared while galois_key is used, and exclusively while
            // keys are added to it.
            std::shared_mutex galois_mutex;
            HEEncoder<Scheme::CKKS> encoder;
            HEEncryptor<Scheme::CKKS> encryptor;
            HEDecryptor<Scheme::CKKS> decryptor;
            HEArithmeticOperator<Scheme::CKKS> operators;
        };

        // shift as a left rotation in [0, slot count).
        static int normalized_shift(int shift, int slot_count)
        {
            return ((shift % slot_count) + slot_count) % slot_count;
        }
    } // namespace detail

    using State = detail::SessionState<Scheme::CKKS>;
    using CKKSCiphertext = SessionCiphertext<Scheme::CKKS>;

    SessionCiphertext<Scheme::CKKS>::SessionCiphertext(
        std::shared_ptr<State> state, Ciphertext<Scheme::CKKS> ciphertext)
        : state_(std::move(state)), ciphertext_(std::move(ciphertext))
    {
    }

    State& SessionCiphertext<Scheme::CKKS>::state() const
    {
        if (!state_)
        {
            throw StateError("The SessionCiphertext is empty! Assign it a "
                             "ciphertext of an HESession first.");
        }
        return *state_;
    }

    void SessionCiphertext<Scheme::CKKS>::check_session(
        const SessionCiphertext& other) const
    {
        other.state();
        if (state_ != other.state_)
        {
            throw ContextMismatchError("The ciphertexts are of different "
                                       "HESession objects!");
        }
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::result(
        Ciphertext<Scheme::CKKS> ciphertext) const
    {
        return CKKSCiphertext(state_, std::move(ciphertext));
    }

    Plaintext<Scheme::CKKS>
    SessionCiphertext<Scheme::CKKS>::encode(const std::vector<double>& values,
                                            bool multiply) const
    {
        State& session = state();
        if (session.context->get_rescale_mode() == rescale_mode::automatic)
        {
            // The level the operator brings the ciphertext to, with the
            // scale of that level.
            int depth =
                ciphertext_.depth() + (ciphertext_.rescale_required() ? 1 : 0);
            return session.encoder.encode(values, depth);
        }
        double scale = multiply ? session.scale : ciphertext_.scale();
        return session.encoder.encode(values, scale, ciphertext_.depth());
    }

    CKKSCiphertext
    SessionCiphertext<Scheme::CKKS>::add(const SessionCiphertext& other) const
    {
        check_session(other);
        return result(state().operators.add(ciphertext_, other.ciphertext_));
    }

    CKKSCiphertext
    SessionCiphertext<Scheme::CKKS>::sub(const SessionCiphertext& other) const
    {
        check_session(other);
        return result(state().operators.sub(ciphertext_, other.ciphertext_));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::multiply(
        const SessionCiphertext& other) const
    {
        check_session(other);
        State& session = state();
        Ciphertext<Scheme::CKKS> product =
            session.operators.multiply(ciphertext_, other.ciphertext_);
        session.operators.relinearize_inplace(product, session.relin_key);
        return result(std::move(product));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::negate() const
    {
        return result(state().operators.negate(ciphertext_));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::add(double value) const
    {
        return result(state().operators.add_plain(ciphertext_, value));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::multiply(double value) const
    {
        // In rescale_mode::automatic, the operator takes the scale of the
        // level instead.
        State& session = state();
        return result(session.operators.multiply_plain(ciphertext_, value,
                                                       session.scale));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::add(
        const std::vector<double>& values) const
    {
        Plaintext<Scheme::CKKS> plain = encode(values, false);
        return result(state().operators.add_plain(ciphertext_, plain));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::sub(
        const std::vector<double>& values) const
    {
        Plaintext<Scheme::CKKS> plain = encode(values, false);
        return result(state().operators.sub_plain(ciphertext_, plain));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::multiply(
        const std::vector<double>& values) const
    {
        Plaintext<Scheme::CKKS> plain = encode(values, true);
        return result(state().operators.multiply_plain(ciphertext_, plain));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::rotate(int shift) const
    {
        State& session = state();
        int slot_count = session.context->get_poly_modulus_degree() / 2;
        int step = detail::normalized_shift(shift, slot_count);
        if (step == 0)
        {
            return *this;
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::CKKS>::rotation_keys(session, {step});
        return result(
            session.operators.rotate(ciphertext_, *session.galois_key, step));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::conjugate() const
    {
        State& session = state();
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::CKKS>::rotation_keys(session, {});
        return result(
            session.operators.conjugate(ciphertext_, *session.galois_key));
    }

    static HEParameterSet<Scheme::CKKS>
    session_parameters(const HEParams<Scheme::CKKS>& params)
    {
        HEParameterSet<Scheme::CKKS> parameters = params.generate();
        parameters.rescale_mode =
            params.rescale_mode.value_or(rescale_mode::automatic);
        return parameters;
    }

    HESession<Scheme::CKKS>::HESession(const HEParams<Scheme::CKKS>& params,
                                       const std::vector<int>& rotations)
        : HESession(session_parameters(params), rotations)
    {
    }

    HESession<Scheme::CKKS>::HESession(
        const HEParameterSet<Scheme::CKKS>& parameters,
        const std::vector<int>& rotations)
        : state_(std::make_shared<State>(parameters))
    {
        if (!rotations.empty())
        {
            generate_rotations(*state_, rotations);
        }
    }

    std::shared_lock<std::shared_mutex>
    HESession<Scheme::CKKS>::rotation_keys(State& state,
                                           const std::vector<int>& shifts)
    {
        int slot_count = state.context->get_poly_modulus_degree() / 2;
        std::shared_lock<std::shared_mutex> lock(state.galois_mutex);
        bool present = state.galois_key.has_value();
        for (int shift : shifts)
        {
            int step = detail::normalized_shift(shift, slot_count);
            present =
                present && ((step == 0) ||
                            (state.galois_key->galois_elt.count(step) != 0));
        }
        if (present)
        {
            return lock;
        }
        lock.unlock();
        {
            std::unique_lock<std::shared_mutex> exclusive(state.galois_mutex);
            generate_rotations(state, shifts);
        }
        // Keys are only added, so the keys of shifts are still there.
        lock.lock();
        return lock;
    }

    void
    HESession<Scheme::CKKS>::generate_rotations(State& state,
                                                const std::vector<int>& shifts)
    {
        int slot_count = state.context->get_poly_modulus_degree() / 2;
        std::vector<int> missing;
        for (int shift : shifts)
        {
            int step = detail::normalized_shift(shift, slot_count);
            bool present = (step == 0) ||
                           (state.galois_key &&
                            state.galois_key->galois_elt.count(step) != 0) ||
                           (std::find(missing.begin(), missing.end(), step) !=
                            missing.end());
            if (!present)
            {
                missing.push_back(step);
            }
        }

        if (!state.galois_key)
        {
            state.galois_key.emplace(state.context, missing);
            state.keygen.generate_galois_key(*state.galois_key,
                                             state.secret_key);
            return;
        }
        if (missing.empty())
        {
            return;
        }

        // The new keys come with a conjugation key of their own, which is
        // dropped: the session keeps the first one.
        Galoiskey<Scheme::CKKS> keys(state.context, missing);
        state.keygen.generate_galois_key(keys, state.secret_key);
        Galoiskey<Scheme::CKKS>& galois_key = *state.galois_key;
        for (int step : missing)
        {
            int galois_elt = keys.galois_elt[step];
            galois_key.galois_elt[step] = galois_elt;
            galois_key.device_location_[galois_elt] =
                std::move(keys.device_location_[galois_elt]);
        }
        galois_key.set_rotation_limits();
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::result(Ciphertext<Scheme::CKKS> ciphertext) const
    {
        return CKKSCiphertext(state_, std::move(ciphertext));
    }

    void
    HESession<Scheme::CKKS>::check_session(const CKKSCiphertext& input) const
    {
        input.state();
        if (input.state_ != state_)
        {
            throw ContextMismatchError("The ciphertext is of another "
                                       "HESession!");
        }
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::encrypt(const std::vector<double>& values) const
    {
        Plaintext<Scheme::CKKS> plain =
            state_->encoder.encode(values, state_->scale);
        return encrypt(plain);
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::encrypt(const std::vector<Complex64>& values) const
    {
        Plaintext<Scheme::CKKS> plain =
            state_->encoder.encode(values, state_->scale);
        return encrypt(plain);
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::encrypt(const Plaintext<Scheme::CKKS>& plain) const
    {
        return result(state_->encryptor.encrypt(plain));
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::wrap(Ciphertext<Scheme::CKKS> ciphertext) const
    {
        ciphertext.validate_context(state_->context);
        return result(std::move(ciphertext));
    }

    std::vector<double>
    HESession<Scheme::CKKS>::decrypt(const CKKSCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::CKKS> plain =
            state_->decryptor.decrypt(input.ciphertext_);
        return state_->encoder.decode(plain);
    }

    std::vector<Complex64>
    HESession<Scheme::CKKS>::decrypt_complex(const CKKSCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::CKKS> plain =
            state_->decryptor.decrypt(input.ciphertext_);
        std::vector<Complex64> values;
        state_->encoder.decode(values, plain);
        return values;
    }

    CKKSCiphertext HESession<Scheme::CKKS>::evaluate_polynomial(
        const CKKSCiphertext& input,
        const std::vector<double>& coefficients) const
    {
        check_session(input);
        return result(state_->operators.evaluate_polynomial(
            input.ciphertext_, coefficients, state_->relin_key));
    }

    CKKSCiphertext HESession<Scheme::CKKS>::evaluate_chebyshev(
        const CKKSCiphertext& input, const std::vector<double>& coefficients,
        double a, double b) const
    {
        check_session(input);
        return result(state_->operators.evaluate_chebyshev(
            input.ciphertext_, coefficients, a, b, state_->relin_key));
    }

    CKKSCiphertext HESession<Scheme::CKKS>::evaluate_chebyshev(
        const CKKSCiphertext& input,
        const std::function<double(double)>& function, double a, double b,
        int degree) const
    {
        check_session(input);
        return result(state_->operators.evaluate_chebyshev(
            input.ciphertext_, function, a, b, degree, state_->relin_key));
    }

    void
    HESession<Scheme::CKKS>::add_rotations(const std::vector<int>& shifts) const
    {
        rotation_keys(*state_, shifts);
    }

    int HESession<Scheme::CKKS>::slot_count() const
    {
        return state_->context->get_poly_modulus_degree() / 2;
    }

    double HESession<Scheme::CKKS>::scale() const
    {
        return state_->scale;
    }

    rescale_mode HESession<Scheme::CKKS>::get_rescale_mode() const
    {
        return state_->context->get_rescale_mode();
    }

    HEContext<Scheme::CKKS> HESession<Scheme::CKKS>::context() const
    {
        return state_->context;
    }

    Secretkey<Scheme::CKKS>& HESession<Scheme::CKKS>::secret_key() const
    {
        return state_->secret_key;
    }

    Publickey<Scheme::CKKS>& HESession<Scheme::CKKS>::public_key() const
    {
        return state_->public_key;
    }

    Relinkey<Scheme::CKKS>& HESession<Scheme::CKKS>::relin_key() const
    {
        return state_->relin_key;
    }

    Galoiskey<Scheme::CKKS>& HESession<Scheme::CKKS>::galois_key() const
    {
        rotation_keys(*state_, {});
        return *state_->galois_key;
    }

    HEEncoder<Scheme::CKKS>& HESession<Scheme::CKKS>::encoder() const
    {
        return state_->encoder;
    }

    HEEncryptor<Scheme::CKKS>& HESession<Scheme::CKKS>::encryptor() const
    {
        return state_->encryptor;
    }

    HEDecryptor<Scheme::CKKS>& HESession<Scheme::CKKS>::decryptor() const
    {
        return state_->decryptor;
    }

    HEArithmeticOperator<Scheme::CKKS>&
    HESession<Scheme::CKKS>::operators() const
    {
        return state_->operators;
    }

} // namespace heongpu
