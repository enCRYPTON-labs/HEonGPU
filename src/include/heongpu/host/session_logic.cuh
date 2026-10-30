// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SESSION_LOGIC_H
#define HEONGPU_SESSION_LOGIC_H

#include <heongpu/host/bfv/session.cuh>
#include <heongpu/host/ckks/session.cuh>

namespace heongpu
{
    /**
     * Logic gates of the bits, 0 or 1, in the slots of the ciphertexts of a
     * BFV or CKKS session. They are computed with the arithmetic of the
     * session, so they also work in the automatic rescale mode:
     * NOT(a) = 1 - a, AND(a, b) = a b, OR(a, b) = a + b - a b,
     * XOR(a, b) = a + b - 2 a b, and NAND, NOR and XNOR are their NOT. Every
     * gate of two inputs uses one multiplication, one level of CKKS. In CKKS
     * the values move away from 0 and 1 with the noise; the bit and gate
     * bootstrappings of a session made from a preset bring them back.
     */
    template <Scheme S> SessionCiphertext<S> NOT(const SessionCiphertext<S>& a)
    {
        return 1 - a;
    }

    template <Scheme S>
    SessionCiphertext<S> AND(const SessionCiphertext<S>& a,
                             const SessionCiphertext<S>& b)
    {
        return a * b;
    }

    template <Scheme S>
    SessionCiphertext<S> OR(const SessionCiphertext<S>& a,
                            const SessionCiphertext<S>& b)
    {
        return a + b - a * b;
    }

    template <Scheme S>
    SessionCiphertext<S> XOR(const SessionCiphertext<S>& a,
                             const SessionCiphertext<S>& b)
    {
        // A sum instead of a product by 2, which would use a level of CKKS.
        SessionCiphertext<S> product = a * b;
        return a + b - (product + product);
    }

    template <Scheme S>
    SessionCiphertext<S> NAND(const SessionCiphertext<S>& a,
                              const SessionCiphertext<S>& b)
    {
        return 1 - a * b;
    }

    template <Scheme S>
    SessionCiphertext<S> NOR(const SessionCiphertext<S>& a,
                             const SessionCiphertext<S>& b)
    {
        return 1 - OR(a, b);
    }

    template <Scheme S>
    SessionCiphertext<S> XNOR(const SessionCiphertext<S>& a,
                              const SessionCiphertext<S>& b)
    {
        return 1 - XOR(a, b);
    }

} // namespace heongpu
#endif // HEONGPU_SESSION_LOGIC_H
