// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/session.cuh>
#include <heongpu/host/bfv/keygenerator.cuh>
#include <algorithm>
#include <optional>

namespace heongpu
{
    namespace detail
    {
        template <> struct SessionState<Scheme::BFV>
        {
            explicit SessionState(const HEParameterSet<Scheme::BFV>& parameters)
                : context(generated_context(parameters)), keygen(context),
                  secret_key(keygen.secret_key()),
                  public_key(keygen.public_key(secret_key)),
                  relin_key(keygen.relin_key(secret_key)), encoder(context),
                  encryptor(context, public_key),
                  decryptor(context, secret_key), operators(context, encoder)
            {
            }

            static HEContext<Scheme::BFV>
            generated_context(const HEParameterSet<Scheme::BFV>& parameters)
            {
                HEContext<Scheme::BFV> context = parameters.make_context();
                context->generate();
                return context;
            }

            // The number of slots in each of the two rows.
            int row_size() const
            {
                return int(context->get_poly_modulus_degree() / 2);
            }

            HEContext<Scheme::BFV> context;
            HEKeyGenerator<Scheme::BFV> keygen;
            Secretkey<Scheme::BFV> secret_key;
            Publickey<Scheme::BFV> public_key;
            Relinkey<Scheme::BFV> relin_key;
            // Created by the first rotation, unless the session is given
            // rotations.
            std::optional<Galoiskey<Scheme::BFV>> galois_key;
            HEEncoder<Scheme::BFV> encoder;
            HEEncryptor<Scheme::BFV> encryptor;
            HEDecryptor<Scheme::BFV> decryptor;
            HEArithmeticOperator<Scheme::BFV> operators;
        };
    } // namespace detail

    using BFVState = detail::SessionState<Scheme::BFV>;
    using BFVCiphertext = SessionCiphertext<Scheme::BFV>;

    SessionCiphertext<Scheme::BFV>::SessionCiphertext(
        std::shared_ptr<BFVState> state, Ciphertext<Scheme::BFV> ciphertext)
        : state_(std::move(state)), ciphertext_(std::move(ciphertext))
    {
    }

    BFVState& SessionCiphertext<Scheme::BFV>::state() const
    {
        if (!state_)
        {
            throw StateError("The SessionCiphertext is empty! Assign it a "
                             "ciphertext of an HESession first.");
        }
        return *state_;
    }

    void SessionCiphertext<Scheme::BFV>::check_session(
        const SessionCiphertext& other) const
    {
        other.state();
        if (state_ != other.state_)
        {
            throw ContextMismatchError("The ciphertexts are of different "
                                       "HESession objects!");
        }
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::result(
        Ciphertext<Scheme::BFV> ciphertext) const
    {
        return BFVCiphertext(state_, std::move(ciphertext));
    }

    Plaintext<Scheme::BFV> SessionCiphertext<Scheme::BFV>::encode(
        const std::vector<std::uint64_t>& values) const
    {
        return state().encoder.encode(values);
    }

    Plaintext<Scheme::BFV> SessionCiphertext<Scheme::BFV>::encode(
        const std::vector<std::int64_t>& values) const
    {
        BFVState& session = state();
        Plaintext<Scheme::BFV> plain(session.context);
        session.encoder.encode(plain, values);
        return plain;
    }

    Plaintext<Scheme::BFV>
    SessionCiphertext<Scheme::BFV>::constant(std::int64_t value) const
    {
        BFVState& session = state();
        return encode(std::vector<std::int64_t>(
            session.context->get_poly_modulus_degree(), value));
    }

    BFVCiphertext
    SessionCiphertext<Scheme::BFV>::add(const SessionCiphertext& other) const
    {
        check_session(other);
        return result(state().operators.add(ciphertext_, other.ciphertext_));
    }

    BFVCiphertext
    SessionCiphertext<Scheme::BFV>::sub(const SessionCiphertext& other) const
    {
        check_session(other);
        return result(state().operators.sub(ciphertext_, other.ciphertext_));
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::multiply(
        const SessionCiphertext& other) const
    {
        check_session(other);
        BFVState& session = state();
        Ciphertext<Scheme::BFV> product =
            session.operators.multiply(ciphertext_, other.ciphertext_);
        session.operators.relinearize_inplace(product, session.relin_key);
        return result(std::move(product));
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::negate() const
    {
        return result(state().operators.negate(ciphertext_));
    }

    BFVCiphertext
    SessionCiphertext<Scheme::BFV>::add(Plaintext<Scheme::BFV> plain) const
    {
        return result(state().operators.add_plain(ciphertext_, plain));
    }

    BFVCiphertext
    SessionCiphertext<Scheme::BFV>::sub(Plaintext<Scheme::BFV> plain) const
    {
        return result(state().operators.sub_plain(ciphertext_, plain));
    }

    BFVCiphertext
    SessionCiphertext<Scheme::BFV>::multiply(Plaintext<Scheme::BFV> plain) const
    {
        return result(state().operators.multiply_plain(ciphertext_, plain));
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::rotate(int shift) const
    {
        BFVState& session = state();
        int row_size = session.row_size();
        // shift as a left rotation in [0, row size).
        int step = ((shift % row_size) + row_size) % row_size;
        if (step == 0)
        {
            return *this;
        }
        HESession<Scheme::BFV>::generate_rotations(session, {step});
        return result(session.operators.rotate_rows(ciphertext_,
                                                    *session.galois_key, step));
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::rotate_columns() const
    {
        BFVState& session = state();
        HESession<Scheme::BFV>::generate_rotations(session, {});
        return result(
            session.operators.rotate_columns(ciphertext_, *session.galois_key));
    }

    HESession<Scheme::BFV>::HESession(const HEParams<Scheme::BFV>& params,
                                      const std::vector<int>& rotations)
        : HESession(params.generate(), rotations)
    {
    }

    HESession<Scheme::BFV>::HESession(
        const HEParameterSet<Scheme::BFV>& parameters,
        const std::vector<int>& rotations)
        : state_(std::make_shared<BFVState>(parameters))
    {
        if (!rotations.empty())
        {
            generate_rotations(*state_, rotations);
        }
    }

    void
    HESession<Scheme::BFV>::generate_rotations(BFVState& state,
                                               const std::vector<int>& shifts)
    {
        int row_size = state.row_size();
        std::vector<int> missing;
        for (int shift : shifts)
        {
            int step = ((shift % row_size) + row_size) % row_size;
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

        // The new keys come with a column rotation key of their own, which
        // is dropped: the session keeps the first one.
        Galoiskey<Scheme::BFV> keys(state.context, missing);
        state.keygen.generate_galois_key(keys, state.secret_key);
        Galoiskey<Scheme::BFV>& galois_key = *state.galois_key;
        for (int step : missing)
        {
            int galois_elt = keys.galois_elt[step];
            galois_key.galois_elt[step] = galois_elt;
            galois_key.device_location_[galois_elt] =
                std::move(keys.device_location_[galois_elt]);
        }
    }

    BFVCiphertext
    HESession<Scheme::BFV>::result(Ciphertext<Scheme::BFV> ciphertext) const
    {
        return BFVCiphertext(state_, std::move(ciphertext));
    }

    void HESession<Scheme::BFV>::check_session(const BFVCiphertext& input) const
    {
        input.state();
        if (input.state_ != state_)
        {
            throw ContextMismatchError("The ciphertext is of another "
                                       "HESession!");
        }
    }

    BFVCiphertext HESession<Scheme::BFV>::encrypt(
        const std::vector<std::uint64_t>& values) const
    {
        Plaintext<Scheme::BFV> plain = state_->encoder.encode(values);
        return encrypt(plain);
    }

    BFVCiphertext HESession<Scheme::BFV>::encrypt(
        const std::vector<std::int64_t>& values) const
    {
        Plaintext<Scheme::BFV> plain(state_->context);
        state_->encoder.encode(plain, values);
        return encrypt(plain);
    }

    BFVCiphertext
    HESession<Scheme::BFV>::encrypt(Plaintext<Scheme::BFV>& plain) const
    {
        return result(state_->encryptor.encrypt(plain));
    }

    BFVCiphertext
    HESession<Scheme::BFV>::wrap(Ciphertext<Scheme::BFV> ciphertext) const
    {
        ciphertext.validate_context(state_->context);
        return result(std::move(ciphertext));
    }

    std::vector<std::uint64_t>
    HESession<Scheme::BFV>::decrypt(const BFVCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::BFV> plain =
            state_->decryptor.decrypt(input.ciphertext_);
        std::vector<std::uint64_t> values;
        state_->encoder.decode(values, plain);
        return values;
    }

    std::vector<std::int64_t>
    HESession<Scheme::BFV>::decrypt_signed(const BFVCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::BFV> plain =
            state_->decryptor.decrypt(input.ciphertext_);
        std::vector<std::int64_t> values;
        state_->encoder.decode(values, plain);
        return values;
    }

    int HESession<Scheme::BFV>::noise_budget(const BFVCiphertext& input) const
    {
        check_session(input);
        return state_->decryptor.remainder_noise_budget(input.ciphertext_);
    }

    void
    HESession<Scheme::BFV>::add_rotations(const std::vector<int>& shifts) const
    {
        generate_rotations(*state_, shifts);
    }

    int HESession<Scheme::BFV>::slot_count() const
    {
        return int(state_->context->get_poly_modulus_degree());
    }

    Data64 HESession<Scheme::BFV>::plain_modulus() const
    {
        return state_->context->get_plain_modulus().value;
    }

    HEContext<Scheme::BFV> HESession<Scheme::BFV>::context() const
    {
        return state_->context;
    }

    Secretkey<Scheme::BFV>& HESession<Scheme::BFV>::secret_key() const
    {
        return state_->secret_key;
    }

    Publickey<Scheme::BFV>& HESession<Scheme::BFV>::public_key() const
    {
        return state_->public_key;
    }

    Relinkey<Scheme::BFV>& HESession<Scheme::BFV>::relin_key() const
    {
        return state_->relin_key;
    }

    Galoiskey<Scheme::BFV>& HESession<Scheme::BFV>::galois_key() const
    {
        generate_rotations(*state_, {});
        return *state_->galois_key;
    }

    HEEncoder<Scheme::BFV>& HESession<Scheme::BFV>::encoder() const
    {
        return state_->encoder;
    }

    HEEncryptor<Scheme::BFV>& HESession<Scheme::BFV>::encryptor() const
    {
        return state_->encryptor;
    }

    HEDecryptor<Scheme::BFV>& HESession<Scheme::BFV>::decryptor() const
    {
        return state_->decryptor;
    }

    HEArithmeticOperator<Scheme::BFV>& HESession<Scheme::BFV>::operators() const
    {
        return state_->operators;
    }

} // namespace heongpu
