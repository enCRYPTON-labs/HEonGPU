// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SCHEMES_H
#define HEONGPU_SCHEMES_H

#include <heongpu/kernel/defines.h>
#include <memory>
#include <utility>

namespace heongpu
{
    enum class Scheme
    {
        BFV,
        CKKS,
        TFHE
    };

    template <Scheme S> class Ciphertext;

    template <Scheme S> class HEContextImpl;
    template <Scheme S> using HEContext = std::shared_ptr<HEContextImpl<S>>;

    template <Scheme S, typename... Args>
    inline HEContext<S> GenHEContext(Args&&... args)
    {
        return std::make_shared<HEContextImpl<S>>(std::forward<Args>(args)...);
    }

    template <Scheme S> class HEDecryptor;

    template <Scheme S> class HEEncoder;

    template <Scheme S> class HEEncryptor;

    template <Scheme S> class Relinkey;

    template <Scheme S> class MultipartyRelinkey;

    template <Scheme S> class Galoiskey;

    template <Scheme S> class MultipartyGaloiskey;

    template <Scheme S> class Switchkey;

    template <Scheme S> class Bootstrappingkey;

    template <Scheme S> class HEKeyGenerator;

    template <Scheme S> class HEOperator;

    template <Scheme S> class HEArithmeticOperator;

    template <Scheme S> class HELogicOperator;

    template <Scheme S> class Plaintext;

    template <Scheme S> class Publickey;

    template <Scheme S> class MultipartyPublickey;

    template <Scheme S> class Secretkey;

    template <Scheme S> class HEMultiPartyManager;

    template <Scheme S> struct HEParams;

    template <Scheme S> struct HEParameterSet;

    template <Scheme S> class HESession;

    template <Scheme S> class SessionCiphertext;

    // Describes the type of encryption scheme to be used.
    enum class scheme_type : std::uint8_t
    {
        // No scheme set; cannot be used for encryption
        none = 0x0,

        // Brakerski/Fan-Vercauteren scheme
        bfv = 0x1,

        // Cheon-Kim-Kim-Song scheme
        ckks = 0x2,

        // Brakerski-Gentry-Vaikuntanathan scheme
        bgv = 0x3,

        // Fast Fully Homomorphic Encryption over the Torus
        tfhe = 0x4
    };

    enum class sec_level_type : std::uint8_t
    {
        // No security level specified.
        none = 0x0,

        // 128 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec128 = 0x1,

        // 192 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec192 = 0x2,

        // 256 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec256 = 0x3
    };

    enum class keyswitching_type : std::uint8_t
    {
        NONE = 0x0,
        // Earlier versions used it for one P prime. It is method II with
        // digits of one prime; data with it loads as KEYSWITCHING_METHOD_II.
        KEYSWITCHING_METHOD_I
        [[deprecated("Every context uses KEYSWITCHING_METHOD_II.")]] = 0x1,
        // Hybrid key switching, used by every context.
        KEYSWITCHING_METHOD_II = 0x2,
    };

    enum class arithmetic_bootstrapping_type : std::uint8_t
    {
        none = 0x0,
        regular = 0x1,
        slim = 0x2,

        NONE [[deprecated("Use arithmetic_bootstrapping_type::none.")]] = none,
        REGULAR_BOOTSTRAPPING
        [[deprecated("Use arithmetic_bootstrapping_type::regular.")]] = regular,
        SLIM_BOOTSTRAPPING
        [[deprecated("Use arithmetic_bootstrapping_type::slim.")]] = slim,
    };

    enum class logic_bootstrapping_type : std::uint8_t
    {
        none = 0x0,
        bit = 0x1, // https://eprint.iacr.org/2024/767.pdf
        gate = 0x2, // https://eprint.iacr.org/2024/767.pdf

        NONE [[deprecated("Use logic_bootstrapping_type::none.")]] = none,
        BIT_BOOTSTRAPPING [[deprecated("Use logic_bootstrapping_type::bit.")]] =
            bit,
        GATE_BOOTSTRAPPING
        [[deprecated("Use logic_bootstrapping_type::gate.")]] = gate,
    };

    // How CKKS ciphertexts are rescaled; see
    // HEContextImpl<Scheme::CKKS>::set_rescale_mode.
    enum class rescale_mode : std::uint8_t
    {
        // The application rescales and brings operands to the same level.
        manual = 0x0,

        // The operators rescale before the operations that need it and bring
        // operands to the same level and scale.
        automatic = 0x1
    };

    enum class encoding : std::uint8_t
    {
        slot = 0x0,
        coefficient = 0x1,

        SLOT [[deprecated("Use encoding::slot.")]] = slot,
        COEFFICIENT [[deprecated("Use encoding::coefficient.")]] = coefficient
    };

    enum class serialization_format : std::uint8_t
    {
        // The fields only, as written by HEonGPU 1.2 and earlier. Readable by
        // every HEonGPU version. Writing it ends with 2.0; reading it stays.
        v1 [[deprecated("Use serialization_format::v2; HEonGPU 2.0 no "
                        "longer writes v1.")]] = 0x1,

        // A header with a magic number, the format version, the object type,
        // the scheme and the context identity before the fields. Readable by
        // HEonGPU 1.3 and later.
        v2 = 0x2
    };

} // namespace heongpu
#endif // HEONGPU_SCHEMES_H
