// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

// HESession<Scheme::BFV> and the operators of SessionCiphertext.

namespace
{
    constexpr auto S = heongpu::Scheme::BFV;
    using Session = heongpu::HESession<S>;
    using u64 = std::uint64_t;

    heongpu::HEParams<S> params(int depth)
    {
        heongpu::HEParams<S> p;
        p.multiplicative_depth = depth;
        return p;
    }

    std::vector<u64> values(int count, u64 seed, u64 t)
    {
        std::vector<u64> result(count);
        u64 state = seed;
        for (int i = 0; i < count; i++)
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            result[i] = (state >> 20) % t;
        }
        return result;
    }

    // f applied to the slots of x and y modulo t.
    std::vector<u64>
    slotwise(const std::vector<u64>& x, const std::vector<u64>& y, u64 t,
             const std::function<__int128(__int128, __int128)>& f)
    {
        std::vector<u64> result(x.size());
        for (std::size_t i = 0; i < x.size(); i++)
        {
            __int128 value = f(x[i], y[i]) % __int128(t);
            result[i] = u64(value < 0 ? value + t : value);
        }
        return result;
    }
} // namespace

TEST(BFVSession, Circuit)
{
    Session session(params(2));
    int n = session.slot_count();
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t), ys = values(n, 2, t);

    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    auto z = x * x * y + 3 * x - y;
    EXPECT_GT(session.noise_budget(z), 0);
    EXPECT_EQ(session.decrypt(z),
              slotwise(xs, ys, t,
                       [&](__int128 a, __int128 b)
                       { return (a * a % t) * b + 3 * a - b; }));
}

TEST(BFVSession, Operators)
{
    Session session(params(2));
    int n = session.slot_count();
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t), ys = values(n, 2, t);
    std::vector<u64> vs = values(n, 3, t);
    std::vector<std::int64_t> ss(n);
    for (int i = 0; i < n; i++)
    {
        ss[i] = std::int64_t(i % 7) - 3;
    }
    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);

    using F = std::function<__int128(__int128, __int128)>;
    auto check = [&](const heongpu::SessionCiphertext<S>& c,
                     const std::vector<u64>& other, const F& f)
    { EXPECT_EQ(session.decrypt(c), slotwise(xs, other, t, f)); };

    check(x + y, ys, [](__int128 a, __int128 b) { return a + b; });
    check(x - y, ys, [](__int128 a, __int128 b) { return a - b; });
    check(x * y, ys, [](__int128 a, __int128 b) { return a * b; });
    check(-x, ys, [](__int128 a, __int128) { return -a; });
    check(x + 5, ys, [](__int128 a, __int128) { return a + 5; });
    check(5 + x, ys, [](__int128 a, __int128) { return a + 5; });
    check(x - 5, ys, [](__int128 a, __int128) { return a - 5; });
    check(5 - x, ys, [](__int128 a, __int128) { return 5 - a; });
    check(x * -2, ys, [](__int128 a, __int128) { return -2 * a; });
    check(7 * x, ys, [](__int128 a, __int128) { return 7 * a; });

    check(x + vs, vs, [](__int128 a, __int128 v) { return a + v; });
    check(vs + x, vs, [](__int128 a, __int128 v) { return a + v; });
    check(x - vs, vs, [](__int128 a, __int128 v) { return a - v; });
    check(vs - x, vs, [](__int128 a, __int128 v) { return v - a; });
    check(x * vs, vs, [](__int128 a, __int128 v) { return a * v; });
    check(vs * x, vs, [](__int128 a, __int128 v) { return v * a; });

    // Signed vectors.
    std::vector<u64> ss_mod(n);
    for (int i = 0; i < n; i++)
    {
        ss_mod[i] = ss[i] < 0 ? u64(ss[i] + std::int64_t(t)) : u64(ss[i]);
    }
    check(x * ss, ss_mod, [](__int128 a, __int128 v) { return a * v; });
    check(x - ss, ss_mod, [](__int128 a, __int128 v) { return a - v; });

    // Compound assignments.
    auto z = x;
    z += y;
    z *= x;
    z -= 4;
    z *= 3;
    z += vs;
    z -= y;
    z *= ss;
    std::vector<u64> expected(n);
    for (int i = 0; i < n; i++)
    {
        __int128 a = xs[i], b = ys[i];
        __int128 value =
            ((((a + b) * a - 4) % t * 3 + vs[i] - b) % t) * ss[i] % t;
        expected[i] = u64(value < 0 ? value + t : value);
    }
    EXPECT_EQ(session.decrypt(z), expected);
}

TEST(BFVSession, Signed_Values)
{
    Session session(params(1));
    int n = session.slot_count();
    std::int64_t half = std::int64_t(session.plain_modulus() / 2);
    std::vector<std::int64_t> xs(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = (std::int64_t(i) * 7919) % (2 * half) - half;
    }
    auto x = session.encrypt(xs);
    EXPECT_EQ(session.decrypt_signed(x), xs);
    std::vector<std::int64_t> negated(n);
    for (int i = 0; i < n; i++)
    {
        negated[i] = -xs[i];
    }
    EXPECT_EQ(session.decrypt_signed(-x), negated);
}

TEST(BFVSession, Same_As_The_Lower_Level_API)
{
    Session session(params(2));
    int n = session.slot_count();
    u64 t = session.plain_modulus();
    auto x = session.encrypt(values(n, 1, t));
    auto y = session.encrypt(values(n, 2, t));
    auto z = x * x * y + x;

    auto& operators = session.operators();
    auto& X = x.ciphertext();
    auto& Y = y.ciphertext();
    auto X2 = operators.multiply(X, X);
    operators.relinearize_inplace(X2, session.relin_key());
    auto X2Y = operators.multiply(X2, Y);
    operators.relinearize_inplace(X2Y, session.relin_key());
    auto R = operators.add(X2Y, X);

    EXPECT_EQ(session.noise_budget(z),
              session.decryptor().remainder_noise_budget(R));
    auto plain = session.decryptor().decrypt(R);
    std::vector<u64> expected;
    session.encoder().decode(expected, plain);
    EXPECT_EQ(session.decrypt(z), expected);
}

TEST(BFVSession, Rotations)
{
    Session session(params(1), {1, -2});
    int n = session.slot_count();
    int row = n / 2;
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t);
    auto x = session.encrypt(xs);

    auto rotated = [&](int shift)
    {
        std::vector<u64> result(n);
        for (int i = 0; i < n; i++)
        {
            int base = (i / row) * row;
            result[i] = xs[base + ((((i % row) + shift) % row) + row) % row];
        }
        return result;
    };
    // Given up front, generated on first use, and their equivalents.
    for (int shift : {1, -2, 3, -5, row - 1, row, 0, 2 * row + 3})
    {
        EXPECT_EQ(session.decrypt(x.rotate(shift)), rotated(shift))
            << "shift " << shift;
    }

    std::vector<u64> swapped(n);
    for (int i = 0; i < n; i++)
    {
        swapped[i] = xs[(i + row) % n];
    }
    EXPECT_EQ(session.decrypt(x.rotate_columns()), swapped);
    session.add_rotations({7});
    EXPECT_EQ(session.decrypt(x.rotate(7)), rotated(7));
}

TEST(BFVSession, Column_Rotation_First)
{
    Session session(params(1));
    int n = session.slot_count();
    int row = n / 2;
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t);
    auto x = session.encrypt(xs);
    EXPECT_EQ(session.decrypt(x.rotate_columns())[0], xs[row]);
    // Then a rotation, added to the same key.
    EXPECT_EQ(session.decrypt(x.rotate(5))[0], xs[5]);
    EXPECT_EQ(session.decrypt(x.rotate_columns())[0], xs[row]);
}

TEST(BFVSession, Noise_Budget)
{
    Session session(params(2));
    int n = session.slot_count();
    u64 t = session.plain_modulus();
    auto x = session.encrypt(values(n, 1, t));
    int fresh = session.noise_budget(x);
    auto square = x * x;
    int after = session.noise_budget(square);
    EXPECT_GT(fresh, after);
    EXPECT_GT(session.noise_budget(square * square), 0);
}

TEST(BFVSession, Outlives_The_Session)
{
    std::optional<Session> session(std::in_place, params(1));
    int n = session->slot_count();
    u64 t = session->plain_modulus();
    std::vector<u64> xs = values(n, 1, t);
    auto x = session->encrypt(xs);
    auto context = session->context();
    heongpu::Secretkey<S> secret_key = session->secret_key();
    session.reset();

    auto z = x * x + 1;
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    auto plain = decryptor.decrypt(z.ciphertext());
    std::vector<u64> result;
    encoder.decode(result, plain);
    EXPECT_EQ(result, slotwise(xs, xs, t,
                               [](__int128 a, __int128) { return a * a + 1; }));
}

TEST(BFVSession, Errors)
{
    Session first(params(1));
    Session second(params(1));
    int n = first.slot_count();
    std::vector<u64> xs(n, 1);
    auto x = first.encrypt(xs);
    auto y = second.encrypt(xs);
    EXPECT_THROW(x + y, heongpu::ContextMismatchError);
    EXPECT_THROW(x * y, heongpu::ContextMismatchError);
    EXPECT_THROW(second.decrypt(x), heongpu::ContextMismatchError);
    EXPECT_THROW(second.noise_budget(x), heongpu::ContextMismatchError);
    // The context identity follows from the parameters.
    Session other(params(2));
    EXPECT_THROW(other.wrap(x.ciphertext()), heongpu::ContextMismatchError);

    // Copies share the session.
    Session copy = first;
    EXPECT_NO_THROW(copy.decrypt(x + first.wrap(x.ciphertext())));

    heongpu::SessionCiphertext<S> empty;
    EXPECT_THROW(x + empty, heongpu::StateError);
    EXPECT_THROW(empty * 2, heongpu::StateError);
    EXPECT_THROW(first.decrypt(empty), heongpu::StateError);
    empty = x;
    EXPECT_NO_THROW(first.decrypt(empty));
}
