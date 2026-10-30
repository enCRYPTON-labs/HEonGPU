// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/session.cuh>
#include <heongpu/host/bfv/keygenerator.cuh>
#include <heongpu/util/serialization_header.h>
#include <algorithm>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

namespace heongpu
{
    namespace detail
    {
        template <> struct SessionState<Scheme::BFV>
        {
            explicit SessionState(const HEParameterSet<Scheme::BFV>& parameters)
                : context(generated_context(parameters)), keygen(context),
                  secret_key(keygen.secret_key()),
                  public_key(keygen.public_key(*secret_key)),
                  relin_key(keygen.relin_key(*secret_key)), encoder(context),
                  encryptor(context, public_key),
                  decryptor(std::in_place, context, *secret_key),
                  operators(context, encoder)
            {
            }

            // A loaded session; HESession::load sets the keys it has.
            SessionState(HEContext<Scheme::BFV> loaded_context,
                         Publickey<Scheme::BFV> loaded_public_key)
                : context(std::move(loaded_context)), keygen(context),
                  public_key(std::move(loaded_public_key)), encoder(context),
                  encryptor(context, public_key), operators(context, encoder)
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

            // The keys and objects a role may not have; these throw for a
            // session without them.
            Secretkey<Scheme::BFV>& secret()
            {
                if (!secret_key)
                {
                    throw StateError(no_secret_key);
                }
                return *secret_key;
            }
            HEDecryptor<Scheme::BFV>& decrypting()
            {
                if (!decryptor)
                {
                    throw StateError(no_secret_key);
                }
                return *decryptor;
            }
            Relinkey<Scheme::BFV>& relin()
            {
                if (!relin_key)
                {
                    throw KeyNotFoundError(
                        "The HESession has no relinearization key! A session "
                        "loaded with the keys of the encryptor does not "
                        "multiply ciphertexts.");
                }
                return *relin_key;
            }
            static constexpr const char* no_secret_key =
                "The HESession has no secret key! A session loaded with the "
                "keys of the evaluator or the encryptor does not decrypt.";

            HEContext<Scheme::BFV> context;
            HEKeyGenerator<Scheme::BFV> keygen;
            // The owner has the secret key, the owner and the evaluator the
            // relinearization key.
            std::optional<Secretkey<Scheme::BFV>> secret_key;
            Publickey<Scheme::BFV> public_key;
            std::optional<Relinkey<Scheme::BFV>> relin_key;
            // Created by the first rotation, unless the session is given
            // rotations.
            std::optional<Galoiskey<Scheme::BFV>> galois_key;
            // Held shared while galois_key is used, and exclusively while
            // keys are added to it.
            std::shared_mutex galois_mutex;
            HEEncoder<Scheme::BFV> encoder;
            HEEncryptor<Scheme::BFV> encryptor;
            std::optional<HEDecryptor<Scheme::BFV>> decryptor;
            HEArithmeticOperator<Scheme::BFV> operators;
        };

        // The error of a session without the secret key for rotations by
        // steps it has no key for.
        static std::string
        missing_rotations_message(const std::vector<int>& steps)
        {
            std::ostringstream message;
            message << "The HESession has no Galois key";
            for (std::size_t i = 0; i < steps.size(); i++)
            {
                message << (i == 0 ? " for the steps " : ", ") << steps[i];
            }
            message << " and no secret key to generate it! Add the steps with "
                       "add_rotations to the session of the owner and save "
                       "it again.";
            return message.str();
        }
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
        session.operators.relinearize_inplace(product, session.relin());
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
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::BFV>::rotation_keys(session, {step});
        return result(session.operators.rotate_rows(ciphertext_,
                                                    *session.galois_key, step));
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::rotate_columns() const
    {
        BFVState& session = state();
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::BFV>::rotation_keys(session, {});
        return result(
            session.operators.rotate_columns(ciphertext_, *session.galois_key));
    }

    std::vector<BFVCiphertext> SessionCiphertext<Scheme::BFV>::rotate_hoisted(
        const std::vector<int>& shifts) const
    {
        BFVState& session = state();
        int row_size = session.row_size();
        std::vector<int> steps;
        for (int shift : shifts)
        {
            steps.push_back(((shift % row_size) + row_size) % row_size);
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::BFV>::rotation_keys(session, steps);
        std::vector<Ciphertext<Scheme::BFV>> rotated =
            session.operators.rotate_rows_hoisted(ciphertext_,
                                                  *session.galois_key, steps);
        std::vector<BFVCiphertext> results;
        results.reserve(rotated.size());
        for (Ciphertext<Scheme::BFV>& ciphertext : rotated)
        {
            results.push_back(result(std::move(ciphertext)));
        }
        return results;
    }

    BFVCiphertext SessionCiphertext<Scheme::BFV>::sum_slots() const
    {
        BFVState& session = state();
        std::vector<int> steps;
        for (int step = 1; step < session.row_size(); step <<= 1)
        {
            steps.push_back(step);
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::BFV>::rotation_keys(session, steps);
        return result(
            session.operators.sum_slots(ciphertext_, *session.galois_key));
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

    std::shared_lock<std::shared_mutex>
    HESession<Scheme::BFV>::rotation_keys(BFVState& state,
                                          const std::vector<int>& shifts)
    {
        int row_size = state.row_size();
        std::shared_lock<std::shared_mutex> lock(state.galois_mutex);
        bool present = state.galois_key.has_value();
        for (int shift : shifts)
        {
            int step = ((shift % row_size) + row_size) % row_size;
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

        if (state.galois_key && missing.empty())
        {
            return;
        }
        if (!state.secret_key)
        {
            throw KeyNotFoundError(detail::missing_rotations_message(missing));
        }
        if (!state.galois_key)
        {
            state.galois_key.emplace(state.context, missing);
            state.keygen.generate_galois_key(*state.galois_key,
                                             *state.secret_key);
            return;
        }

        // The new keys come with a column rotation key of their own, which
        // is dropped: the session keeps the first one.
        Galoiskey<Scheme::BFV> keys(state.context, missing);
        state.keygen.generate_galois_key(keys, *state.secret_key);
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
    HESession<Scheme::BFV>::encrypt(const Plaintext<Scheme::BFV>& plain) const
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
            state_->decrypting().decrypt(input.ciphertext_);
        std::vector<std::uint64_t> values;
        state_->encoder.decode(values, plain);
        return values;
    }

    std::vector<std::int64_t>
    HESession<Scheme::BFV>::decrypt_signed(const BFVCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::BFV> plain =
            state_->decrypting().decrypt(input.ciphertext_);
        std::vector<std::int64_t> values;
        state_->encoder.decode(values, plain);
        return values;
    }

    int HESession<Scheme::BFV>::noise_budget(const BFVCiphertext& input) const
    {
        check_session(input);
        return state_->decrypting().remainder_noise_budget(input.ciphertext_);
    }

    void
    HESession<Scheme::BFV>::add_rotations(const std::vector<int>& shifts) const
    {
        rotation_keys(*state_, shifts);
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
        return state_->secret();
    }

    Publickey<Scheme::BFV>& HESession<Scheme::BFV>::public_key() const
    {
        return state_->public_key;
    }

    Relinkey<Scheme::BFV>& HESession<Scheme::BFV>::relin_key() const
    {
        return state_->relin();
    }

    Galoiskey<Scheme::BFV>& HESession<Scheme::BFV>::galois_key() const
    {
        rotation_keys(*state_, {});
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
        return state_->decrypting();
    }

    HEArithmeticOperator<Scheme::BFV>& HESession<Scheme::BFV>::operators() const
    {
        return state_->operators;
    }

    namespace
    {
        constexpr const char* session_name = "HESession";
        // The version of the fields after the header.
        constexpr std::uint8_t session_version = 1;

        const char* role_name(session_role role)
        {
            switch (role)
            {
                case session_role::owner:
                    return "owner";
                case session_role::evaluator:
                    return "evaluator";
                case session_role::encryptor:
                    return "encryptor";
            }
            return "unknown role";
        }

        // A key of context read from is, checked to be of context.
        template <typename Key>
        void load_key(std::optional<Key>& key,
                      const HEContext<Scheme::BFV>& context, std::istream& is)
        {
            key.emplace(context);
            key->load(is);
            key->store_in_device();
        }
    } // namespace

    HESession<Scheme::BFV>::HESession(std::shared_ptr<BFVState> state)
        : state_(std::move(state))
    {
    }

    void HESession<Scheme::BFV>::save(std::ostream& os, session_role role) const
    {
        BFVState& state = *state_;
        if ((role != session_role::owner) &&
            (role != session_role::evaluator) &&
            (role != session_role::encryptor))
        {
            throw ParameterError(error_message("Invalid session role ",
                                               static_cast<int>(role), "!"));
        }
        if ((role == session_role::owner) && !state.secret_key)
        {
            throw StateError("The HESession has no secret key to save with "
                             "the keys of the owner!");
        }
        if ((role == session_role::evaluator) && !state.relin_key)
        {
            throw StateError("The HESession has no evaluation keys to save "
                             "with the keys of the evaluator!");
        }

        detail::write_serialization_header(
            os, serialization_format::v2, detail::serialized_object::session,
            scheme_type::bfv, state.context->context_id());
        detail::write_field(os, session_version);
        detail::write_field(os, role);
        state.context->save(os);
        state.public_key.save(os);
        if (role != session_role::encryptor)
        {
            state.relin_key->save(os);
            std::shared_lock<std::shared_mutex> lock(state.galois_mutex);
            detail::write_field(
                os, static_cast<std::uint8_t>(state.galois_key.has_value()));
            if (state.galois_key)
            {
                state.galois_key->save(os);
            }
        }
        if (role == session_role::owner)
        {
            state.secret_key->save(os);
        }
        if (!os)
        {
            throw SerializationError("Writing the HESession failed!");
        }
    }

    HESession<Scheme::BFV> HESession<Scheme::BFV>::load(std::istream& is)
    {
        return load_session(is, nullptr);
    }

    HESession<Scheme::BFV> HESession<Scheme::BFV>::load(std::istream& is,
                                                        session_role expected)
    {
        return load_session(is, &expected);
    }

    HESession<Scheme::BFV>
    HESession<Scheme::BFV>::load_session(std::istream& is,
                                         const session_role* expected)
    {
        if (is.peek() != 'H')
        {
            throw SerializationError("Invalid HESession binary! It does not "
                                     "start with the HEGP header.");
        }
        const std::uint64_t context_id = detail::read_serialization_header(
            is, detail::serialized_object::session, scheme_type::bfv);
        auto version = detail::read_field<std::uint8_t>(is, session_name);
        if (version != session_version)
        {
            throw SerializationError(error_message(
                "Unsupported HESession binary version ", int(version),
                "! This HEonGPU version reads the version ",
                int(session_version), "."));
        }
        auto role = detail::read_field<session_role>(is, session_name);
        if (static_cast<std::uint8_t>(role) >
            static_cast<std::uint8_t>(session_role::encryptor))
        {
            throw SerializationError(error_message(
                "Invalid HESession binary! Unknown role ", int(role), "."));
        }
        if (expected && (role != *expected))
        {
            throw StateError(
                error_message("The HESession was saved with the keys of the ",
                              role_name(role), ", expected those of the ",
                              role_name(*expected), "!"));
        }

        HEContext<Scheme::BFV> context = GenHEContext<Scheme::BFV>();
        context->load(is);
        if (context->context_id() != context_id)
        {
            throw SerializationError("Corrupted HESession binary! The context "
                                     "does not have the identity of the "
                                     "header.");
        }
        std::optional<Publickey<Scheme::BFV>> public_key;
        load_key(public_key, context, is);
        auto state =
            std::make_shared<BFVState>(context, std::move(*public_key));
        if (role != session_role::encryptor)
        {
            load_key(state->relin_key, context, is);
            if (detail::read_field<std::uint8_t>(is, session_name) != 0)
            {
                load_key(state->galois_key, context, is);
            }
        }
        if (role == session_role::owner)
        {
            load_key(state->secret_key, context, is);
            state->decryptor.emplace(context, *state->secret_key);
        }
        return HESession(std::move(state));
    }

    BFVCiphertext
    HESession<Scheme::BFV>::load_ciphertext(std::istream& is) const
    {
        Ciphertext<Scheme::BFV> ciphertext(state_->context);
        ciphertext.load(is);
        return wrap(std::move(ciphertext));
    }

    session_role HESession<Scheme::BFV>::role() const
    {
        if (state_->secret_key)
        {
            return session_role::owner;
        }
        return state_->relin_key ? session_role::evaluator
                                 : session_role::encryptor;
    }

} // namespace heongpu
