// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/session.cuh>
#include <heongpu/host/ckks/keygenerator.cuh>
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
        template <> struct SessionState<Scheme::CKKS>
        {
            explicit SessionState(
                const HEParameterSet<Scheme::CKKS>& parameters)
                : context(generated_context(parameters)),
                  scale(parameters.scale), keygen(context),
                  secret_key(keygen.secret_key()),
                  public_key(keygen.public_key(*secret_key)),
                  relin_key(keygen.relin_key(*secret_key)), encoder(context),
                  encryptor(context, public_key),
                  decryptor(std::in_place, context, *secret_key),
                  operators(context, encoder)
            {
            }

            // The bootstrapping keys are generated up front; the
            // relinearization and Galois keys of the session are those of
            // the bootstrapping.
            explicit SessionState(const CKKSBootstrappingPreset& bootstrapping)
                : context(preset_context(bootstrapping)),
                  scale(bootstrapping.scale), keygen(context),
                  secret_key(preset_secret_key(keygen, context, bootstrapping)),
                  public_key(keygen.public_key(*secret_key)), encoder(context),
                  encryptor(context, public_key),
                  decryptor(std::in_place, context, *secret_key),
                  operators(context, encoder)
            {
                std::vector<int> indices = prepare_bootstrapping(bootstrapping);
                CKKSBootstrappingKeys keys = keygen.bootstrapping_keys(
                    *secret_key, indices,
                    bootstrapping.sparse_secret_key_hamming_weight);
                relin_key = std::move(keys.relin_key);
                galois_key.emplace(std::move(keys.galois_key));
                dense_to_sparse = std::move(keys.dense_to_sparse);
                sparse_to_dense = std::move(keys.sparse_to_dense);
            }

            // A loaded session; HESession::load sets the keys it has.
            SessionState(HEContext<Scheme::CKKS> loaded_context,
                         double loaded_scale,
                         Publickey<Scheme::CKKS> loaded_public_key)
                : context(std::move(loaded_context)), scale(loaded_scale),
                  keygen(context), public_key(std::move(loaded_public_key)),
                  encoder(context), encryptor(context, public_key),
                  operators(context, encoder)
            {
            }

            // Generates the bootstrapping parameters of bootstrapping and
            // returns the shifts of its Galois keys.
            std::vector<int>
            prepare_bootstrapping(const CKKSBootstrappingPreset& bootstrapping)
            {
                preset = bootstrapping;
                if (bootstrapping.arithmetic_type !=
                    arithmetic_bootstrapping_type::none)
                {
                    operators.generate_bootstrapping_params(bootstrapping);
                    return operators.bootstrapping_key_indices();
                }
                logic.emplace(context, encoder, bootstrapping.scale);
                logic->generate_bootstrapping_params(bootstrapping);
                return logic->bootstrapping_key_indices();
            }

            static HEContext<Scheme::CKKS>
            generated_context(const HEParameterSet<Scheme::CKKS>& parameters)
            {
                HEContext<Scheme::CKKS> context = parameters.make_context();
                context->generate();
                return context;
            }

            // The context of preset, whose top level in
            // rescale_mode::automatic is the level of the bootstrapping
            // results.
            static HEContext<Scheme::CKKS>
            preset_context(const CKKSBootstrappingPreset& preset)
            {
                HEContext<Scheme::CKKS> context =
                    GenHEContext<Scheme::CKKS>(sec_level_type::sec128);
                preset.apply(context);
                context->generate();
                context->set_rescale_mode(rescale_mode::automatic, preset.scale,
                                          preset.levels().output_level);
                return context;
            }

            static Secretkey<Scheme::CKKS>
            preset_secret_key(HEKeyGenerator<Scheme::CKKS>& keygen,
                              const HEContext<Scheme::CKKS>& context,
                              const CKKSBootstrappingPreset& preset)
            {
                Secretkey<Scheme::CKKS> secret_key(
                    context, preset.secret_key_hamming_weight);
                keygen.generate_secret_key(secret_key);
                return secret_key;
            }

            // The keys and objects a role may not have; these throw for a
            // session without them.
            Secretkey<Scheme::CKKS>& secret()
            {
                if (!secret_key)
                {
                    throw StateError(no_secret_key);
                }
                return *secret_key;
            }
            HEDecryptor<Scheme::CKKS>& decrypting()
            {
                if (!decryptor)
                {
                    throw StateError(no_secret_key);
                }
                return *decryptor;
            }
            Relinkey<Scheme::CKKS>& relin()
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

            // The switch keys of the bootstrapping, or null.
            Switchkey<Scheme::CKKS>* dense_to_sparse_key()
            {
                return dense_to_sparse ? &*dense_to_sparse : nullptr;
            }
            Switchkey<Scheme::CKKS>* sparse_to_dense_key()
            {
                return sparse_to_dense ? &*sparse_to_dense : nullptr;
            }

            HEContext<Scheme::CKKS> context;
            double scale;
            HEKeyGenerator<Scheme::CKKS> keygen;
            // The owner has the secret key, the owner and the evaluator the
            // relinearization key.
            std::optional<Secretkey<Scheme::CKKS>> secret_key;
            Publickey<Scheme::CKKS> public_key;
            std::optional<Relinkey<Scheme::CKKS>> relin_key;
            // Created by the first rotation or conjugation, unless the
            // session is given rotations.
            std::optional<Galoiskey<Scheme::CKKS>> galois_key;
            // Held shared while galois_key is used, and exclusively while
            // keys are added to it.
            std::shared_mutex galois_mutex;
            HEEncoder<Scheme::CKKS> encoder;
            HEEncryptor<Scheme::CKKS> encryptor;
            std::optional<HEDecryptor<Scheme::CKKS>> decryptor;
            HEArithmeticOperator<Scheme::CKKS> operators;
            // The bootstrapping of a session made from a preset; logic for
            // the bit and gate bootstrappings.
            std::optional<CKKSBootstrappingPreset> preset;
            std::optional<HELogicOperator<Scheme::CKKS>> logic;
            std::optional<Switchkey<Scheme::CKKS>> dense_to_sparse;
            std::optional<Switchkey<Scheme::CKKS>> sparse_to_dense;
        };

        // values encoded at the top level with its scale.
        template <typename T>
        static Plaintext<Scheme::CKKS>
        top_level_plaintext(SessionState<Scheme::CKKS>& state,
                            const std::vector<T>& values)
        {
            if (state.context->get_rescale_mode() == rescale_mode::automatic)
            {
                return state.encoder.encode(values);
            }
            return state.encoder.encode(values, state.scale);
        }

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
        session.operators.relinearize_inplace(product, session.relin());
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

    std::vector<CKKSCiphertext> SessionCiphertext<Scheme::CKKS>::rotate_hoisted(
        const std::vector<int>& shifts) const
    {
        State& session = state();
        int slot_count = session.context->get_poly_modulus_degree() / 2;
        std::vector<int> steps;
        for (int shift : shifts)
        {
            steps.push_back(detail::normalized_shift(shift, slot_count));
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::CKKS>::rotation_keys(session, steps);
        std::vector<Ciphertext<Scheme::CKKS>> rotated =
            session.operators.rotate_hoisted(ciphertext_, *session.galois_key,
                                             steps);
        std::vector<CKKSCiphertext> results;
        results.reserve(rotated.size());
        for (Ciphertext<Scheme::CKKS>& ciphertext : rotated)
        {
            results.push_back(result(std::move(ciphertext)));
        }
        return results;
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::sum_slots() const
    {
        State& session = state();
        std::vector<int> steps;
        for (int step = 1;
             step < session.context->get_poly_modulus_degree() / 2; step <<= 1)
        {
            steps.push_back(step);
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::CKKS>::rotation_keys(session, steps);
        return result(
            session.operators.sum_slots(ciphertext_, *session.galois_key));
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::inner_product(
        const SessionCiphertext& other) const
    {
        return multiply(other).sum_slots();
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::inner_product(
        const std::vector<double>& values) const
    {
        return multiply(values).sum_slots();
    }

    int SessionCiphertext<Scheme::CKKS>::levels_left() const
    {
        State& session = state();
        int levels =
            ciphertext_.level() - (ciphertext_.rescale_required() ? 1 : 0);
        if (session.preset)
        {
            levels -= session.preset->levels().input_level;
        }
        return levels;
    }

    CKKSCiphertext SessionCiphertext<Scheme::CKKS>::bootstrap() const
    {
        State& session = state();
        if (!session.preset)
        {
            throw StateError("The HESession has no bootstrapping! Create it "
                             "from a CKKSBootstrappingPreset.");
        }
        if (session.preset->logic_type == logic_bootstrapping_type::gate)
        {
            throw StateError("A gate bootstrapping takes two ciphertexts! Use "
                             "HESession::AND_bootstrapping or another gate.");
        }
        std::shared_lock<std::shared_mutex> lock =
            HESession<Scheme::CKKS>::rotation_keys(session, {});
        const ExecutionOptions options;
        switch (session.preset->arithmetic_type)
        {
            case arithmetic_bootstrapping_type::regular:
                return result(session.operators.regular_bootstrapping_ckks(
                    ciphertext_, *session.galois_key, session.relin(),
                    session.dense_to_sparse_key(),
                    session.sparse_to_dense_key(), options));
            case arithmetic_bootstrapping_type::slim:
                return result(session.operators.slim_bootstrapping_ckks(
                    ciphertext_, *session.galois_key, session.relin(),
                    session.dense_to_sparse_key(),
                    session.sparse_to_dense_key(), options));
            default:
                return result(session.logic->bit_bootstrapping_ckks(
                    ciphertext_, *session.galois_key, session.relin(),
                    session.dense_to_sparse_key(),
                    session.sparse_to_dense_key(), options));
        }
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

    HESession<Scheme::CKKS>::HESession(const CKKSBootstrappingPreset& preset,
                                       const std::vector<int>& rotations)
        : state_(std::make_shared<State>(preset))
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

        // The new keys come with a conjugation key of their own, which is
        // dropped: the session keeps the first one.
        Galoiskey<Scheme::CKKS> keys(state.context, missing);
        state.keygen.generate_galois_key(keys, *state.secret_key);
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
        return encrypt(detail::top_level_plaintext(*state_, values));
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::encrypt(const std::vector<Complex64>& values) const
    {
        return encrypt(detail::top_level_plaintext(*state_, values));
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
            state_->decrypting().decrypt(input.ciphertext_);
        return state_->encoder.decode(plain);
    }

    std::vector<Complex64>
    HESession<Scheme::CKKS>::decrypt_complex(const CKKSCiphertext& input) const
    {
        check_session(input);
        Plaintext<Scheme::CKKS> plain =
            state_->decrypting().decrypt(input.ciphertext_);
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
            input.ciphertext_, coefficients, state_->relin()));
    }

    CKKSCiphertext HESession<Scheme::CKKS>::evaluate_chebyshev(
        const CKKSCiphertext& input, const std::vector<double>& coefficients,
        double a, double b) const
    {
        check_session(input);
        return result(state_->operators.evaluate_chebyshev(
            input.ciphertext_, coefficients, a, b, state_->relin()));
    }

    CKKSCiphertext HESession<Scheme::CKKS>::evaluate_chebyshev(
        const CKKSCiphertext& input,
        const std::function<double(double)>& function, double a, double b,
        int degree) const
    {
        check_session(input);
        return result(state_->operators.evaluate_chebyshev(
            input.ciphertext_, function, a, b, degree, state_->relin()));
    }

    CKKSCiphertext HESession<Scheme::CKKS>::gate_bootstrapping(
        int gate, const CKKSCiphertext& input1,
        const CKKSCiphertext& input2) const
    {
        check_session(input1);
        check_session(input2);
        if (!state_->preset ||
            (state_->preset->logic_type != logic_bootstrapping_type::gate))
        {
            throw StateError("The gate bootstrappings need an HESession made "
                             "from a gate bootstrapping preset!");
        }
        std::shared_lock<std::shared_mutex> lock = rotation_keys(*state_, {});
        return result(state_->logic->gate_bootstrapping(
            static_cast<HELogicOperator<Scheme::CKKS>::logic_gate>(gate),
            input1.ciphertext_, input2.ciphertext_, *state_->galois_key,
            state_->relin(), state_->dense_to_sparse_key(),
            state_->sparse_to_dense_key(), ExecutionOptions()));
    }

    // The gates in the order of HELogicOperator<Scheme::CKKS>::logic_gate.
    CKKSCiphertext HESession<Scheme::CKKS>::AND_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(0, input1, input2);
    }

    CKKSCiphertext HESession<Scheme::CKKS>::OR_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(1, input1, input2);
    }

    CKKSCiphertext HESession<Scheme::CKKS>::XOR_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(2, input1, input2);
    }

    CKKSCiphertext HESession<Scheme::CKKS>::NAND_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(3, input1, input2);
    }

    CKKSCiphertext HESession<Scheme::CKKS>::NOR_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(4, input1, input2);
    }

    CKKSCiphertext HESession<Scheme::CKKS>::XNOR_bootstrapping(
        const CKKSCiphertext& input1, const CKKSCiphertext& input2) const
    {
        return gate_bootstrapping(5, input1, input2);
    }

    bool HESession<Scheme::CKKS>::has_bootstrapping() const
    {
        return state_->preset.has_value();
    }

    BootstrappingLevels HESession<Scheme::CKKS>::bootstrapping_levels() const
    {
        if (!state_->preset)
        {
            throw StateError("The HESession has no bootstrapping! Create it "
                             "from a CKKSBootstrappingPreset.");
        }
        return state_->preset->levels();
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
        return state_->secret();
    }

    Publickey<Scheme::CKKS>& HESession<Scheme::CKKS>::public_key() const
    {
        return state_->public_key;
    }

    Relinkey<Scheme::CKKS>& HESession<Scheme::CKKS>::relin_key() const
    {
        return state_->relin();
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
        return state_->decrypting();
    }

    HEArithmeticOperator<Scheme::CKKS>&
    HESession<Scheme::CKKS>::operators() const
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

        void write_matrix_config(std::ostream& os,
                                 const EncodingMatrixConfig& config)
        {
            detail::write_field(os, config.lt_type_);
            detail::write_field(os, config.level_start_);
            detail::write_field(os, config.bsgs_ratio_);
            detail::write_field(os, config.piece_);
        }

        EncodingMatrixConfig read_matrix_config(std::istream& is)
        {
            EncodingMatrixConfig config;
            config.lt_type_ =
                detail::read_field<LinearTransformType>(is, session_name);
            config.level_start_ = detail::read_field<int>(is, session_name);
            config.bsgs_ratio_ = detail::read_field<double>(is, session_name);
            config.piece_ = detail::read_field<int>(is, session_name);
            return config;
        }

        void write_eval_mod_config(std::ostream& os,
                                   const EvalModConfig& config)
        {
            detail::write_field(os, config.Q_);
            detail::write_field(os, config.level_start_);
            detail::write_field(os, config.message_ratio_);
            detail::write_field(os, config.K_);
            detail::write_field(os, config.sine_deg_);
            detail::write_field(os, config.double_angle_);
            detail::write_field(os, config.arcsine_deg_);
            detail::write_field(os, config.scaling_factor_);
            detail::write_field(os, config.piece_);
            detail::write_field(os, config.sqrt2pi_);
            detail::write_field(os, config.q_diff_);
            detail::write_field(os, config.sine_type_);
            detail::write_field(os, config.poly_type_);
        }

        EvalModConfig read_eval_mod_config(std::istream& is)
        {
            EvalModConfig config;
            config.Q_ = detail::read_field<Data64>(is, session_name);
            config.level_start_ = detail::read_field<int>(is, session_name);
            config.message_ratio_ =
                detail::read_field<double>(is, session_name);
            config.K_ = detail::read_field<int>(is, session_name);
            config.sine_deg_ = detail::read_field<int>(is, session_name);
            config.double_angle_ = detail::read_field<int>(is, session_name);
            config.arcsine_deg_ = detail::read_field<int>(is, session_name);
            config.scaling_factor_ =
                detail::read_field<double>(is, session_name);
            config.piece_ = detail::read_field<int>(is, session_name);
            config.sqrt2pi_ = detail::read_field<double>(is, session_name);
            config.q_diff_ = detail::read_field<double>(is, session_name);
            config.sine_type_ = detail::read_field<SineType>(is, session_name);
            config.poly_type_ = detail::read_field<PolyType>(is, session_name);
            return config;
        }

        void write_preset(std::ostream& os,
                          const CKKSBootstrappingPreset& preset)
        {
            detail::write_field(os, preset.arithmetic_type);
            detail::write_field(os, preset.logic_type);
            detail::write_field(
                os, static_cast<std::uint64_t>(preset.poly_modulus_degree));
            detail::write_field(os, preset.coeff_modulus);
            detail::write_field(os, preset.special_modulus);
            detail::write_field(os, preset.coeff_modulus_bit_sizes);
            detail::write_field(os, preset.special_modulus_bit_sizes);
            detail::write_field(os, preset.scale);
            detail::write_field(os, preset.secret_key_hamming_weight);
            detail::write_field(os, preset.sparse_secret_key_hamming_weight);
            detail::write_field(os, preset.config.CtoS_piece_);
            detail::write_field(os, preset.config.StoC_piece_);
            write_matrix_config(os, preset.config.cts_config_);
            write_matrix_config(os, preset.config.stc_config_);
            write_eval_mod_config(os, preset.config.eval_mod_config_);
        }

        CKKSBootstrappingPreset read_preset(std::istream& is)
        {
            CKKSBootstrappingPreset preset;
            preset.arithmetic_type =
                detail::read_field<arithmetic_bootstrapping_type>(is,
                                                                  session_name);
            preset.logic_type =
                detail::read_field<logic_bootstrapping_type>(is, session_name);
            preset.poly_modulus_degree =
                detail::read_field<std::uint64_t>(is, session_name);
            preset.coeff_modulus =
                detail::read_vector_field<Data64>(is, session_name);
            preset.special_modulus =
                detail::read_vector_field<Data64>(is, session_name);
            preset.coeff_modulus_bit_sizes =
                detail::read_vector_field<int>(is, session_name);
            preset.special_modulus_bit_sizes =
                detail::read_vector_field<int>(is, session_name);
            preset.scale = detail::read_field<double>(is, session_name);
            preset.secret_key_hamming_weight =
                detail::read_field<int>(is, session_name);
            preset.sparse_secret_key_hamming_weight =
                detail::read_field<int>(is, session_name);
            preset.config.CtoS_piece_ =
                detail::read_field<int>(is, session_name);
            preset.config.StoC_piece_ =
                detail::read_field<int>(is, session_name);
            preset.config.cts_config_ = read_matrix_config(is);
            preset.config.stc_config_ = read_matrix_config(is);
            preset.config.eval_mod_config_ = read_eval_mod_config(is);
            return preset;
        }

        // A key of context read from is, checked to be of context.
        template <typename Key>
        void load_key(std::optional<Key>& key,
                      const HEContext<Scheme::CKKS>& context, std::istream& is)
        {
            key.emplace(context);
            key->load(is);
            key->store_in_device();
        }
    } // namespace

    HESession<Scheme::CKKS>::HESession(std::shared_ptr<State> state)
        : state_(std::move(state))
    {
    }

    void HESession<Scheme::CKKS>::save(std::ostream& os,
                                       session_role role) const
    {
        State& state = *state_;
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
        bool evaluation = (role != session_role::encryptor);
        bool bootstrapping = evaluation && state.preset.has_value();

        detail::write_serialization_header(
            os, serialization_format::v2, detail::serialized_object::session,
            scheme_type::ckks, state.context->context_id());
        detail::write_field(os, session_version);
        detail::write_field(os, role);
        detail::write_field(os, state.context->get_rescale_mode());
        detail::write_field(os, state.scale);
        detail::write_field(
            os, static_cast<std::int32_t>(state.context->get_top_level()));
        detail::write_field(os, static_cast<std::uint8_t>(bootstrapping));
        if (bootstrapping)
        {
            write_preset(os, *state.preset);
        }

        state.context->save(os);
        state.public_key.save(os);
        if (evaluation)
        {
            state.relin_key->save(os);
            std::shared_lock<std::shared_mutex> lock(state.galois_mutex);
            detail::write_field(
                os, static_cast<std::uint8_t>(state.galois_key.has_value()));
            if (state.galois_key)
            {
                state.galois_key->save(os);
            }
            if (bootstrapping)
            {
                detail::write_field(os, static_cast<std::uint8_t>(
                                            state.dense_to_sparse.has_value()));
                if (state.dense_to_sparse)
                {
                    state.dense_to_sparse->save(os);
                    state.sparse_to_dense->save(os);
                }
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

    HESession<Scheme::CKKS> HESession<Scheme::CKKS>::load(std::istream& is)
    {
        return load_session(is, nullptr);
    }

    HESession<Scheme::CKKS> HESession<Scheme::CKKS>::load(std::istream& is,
                                                          session_role expected)
    {
        return load_session(is, &expected);
    }

    HESession<Scheme::CKKS>
    HESession<Scheme::CKKS>::load_session(std::istream& is,
                                          const session_role* expected)
    {
        if (is.peek() != 'H')
        {
            throw SerializationError("Invalid HESession binary! It does not "
                                     "start with the HEGP header.");
        }
        const std::uint64_t context_id = detail::read_serialization_header(
            is, detail::serialized_object::session, scheme_type::ckks);
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
        auto mode = detail::read_field<rescale_mode>(is, session_name);
        double scale = detail::read_field<double>(is, session_name);
        int top_level = detail::read_field<std::int32_t>(is, session_name);
        std::optional<CKKSBootstrappingPreset> preset;
        if (detail::read_field<std::uint8_t>(is, session_name) != 0)
        {
            preset = read_preset(is);
        }

        HEContext<Scheme::CKKS> context = GenHEContext<Scheme::CKKS>();
        context->load(is);
        if (context->context_id() != context_id)
        {
            throw SerializationError("Corrupted HESession binary! The context "
                                     "does not have the identity of the "
                                     "header.");
        }
        if (mode == rescale_mode::automatic)
        {
            context->set_rescale_mode(mode, scale, top_level);
        }
        std::optional<Publickey<Scheme::CKKS>> public_key;
        load_key(public_key, context, is);
        auto state =
            std::make_shared<State>(context, scale, std::move(*public_key));

        if (role != session_role::encryptor)
        {
            load_key(state->relin_key, context, is);
            if (detail::read_field<std::uint8_t>(is, session_name) != 0)
            {
                load_key(state->galois_key, context, is);
            }
            if (preset)
            {
                state->prepare_bootstrapping(*preset);
                if (detail::read_field<std::uint8_t>(is, session_name) != 0)
                {
                    load_key(state->dense_to_sparse, context, is);
                    load_key(state->sparse_to_dense, context, is);
                }
            }
        }
        if (role == session_role::owner)
        {
            load_key(state->secret_key, context, is);
            state->decryptor.emplace(context, *state->secret_key);
        }
        return HESession(std::move(state));
    }

    CKKSCiphertext
    HESession<Scheme::CKKS>::load_ciphertext(std::istream& is) const
    {
        Ciphertext<Scheme::CKKS> ciphertext(state_->context);
        ciphertext.load(is);
        return wrap(std::move(ciphertext));
    }

    session_role HESession<Scheme::CKKS>::role() const
    {
        if (state_->secret_key)
        {
            return session_role::owner;
        }
        return state_->relin_key ? session_role::evaluator
                                 : session_role::encryptor;
    }

} // namespace heongpu
