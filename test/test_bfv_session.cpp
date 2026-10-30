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
#include <string>
#include <thread>
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

TEST(BFVSession, Threads_Rotate_By_New_Steps)
{
    // Copies of a session share its Galois key; the threads add the keys of
    // their steps to it while the others rotate with it.
    Session session(params(1));
    int n = session.slot_count();
    int row = n / 2;
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t);
    auto x = session.encrypt(xs);

    constexpr int thread_count = 8;
    constexpr int rounds = 3;
    std::vector<std::string> failures(thread_count);
    std::vector<std::thread> threads;
    for (int k = 0; k < thread_count; k++)
    {
        threads.emplace_back(
            [&, k, copy = session]()
            {
                try
                {
                    for (int r = 0; r < rounds; r++)
                    {
                        int shift = 1 + k + thread_count * r;
                        std::vector<u64> result = copy.decrypt(x.rotate(shift));
                        for (int i = 0; i < n; i++)
                        {
                            int base = (i / row) * row;
                            if (result[i] != xs[base + (i % row + shift) % row])
                            {
                                failures[k] = "wrong slot " +
                                              std::to_string(i) + " of shift " +
                                              std::to_string(shift);
                                return;
                            }
                        }
                    }
                }
                catch (const std::exception& error)
                {
                    failures[k] = error.what();
                }
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (int k = 0; k < thread_count; k++)
    {
        EXPECT_EQ(failures[k], "") << "thread " << k;
    }
}

TEST(BFVSession, Sums_Hoisted_Rotations_And_Logic)
{
    Session session(params(1));
    int n = session.slot_count();
    u64 t = session.plain_modulus();
    std::vector<u64> xs = values(n, 1, t);
    std::vector<u64> ys = values(n, 2, t);
    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    unsigned __int128 sum = 0, dot = 0;
    for (int i = 0; i < n; i++)
    {
        sum += xs[i];
        dot += (unsigned __int128) xs[i] * ys[i];
    }

    std::vector<u64> sums = session.decrypt(x.sum_slots());
    EXPECT_EQ(sums[0], u64(sum % t));
    EXPECT_EQ(sums[n - 1], u64(sum % t));
    EXPECT_EQ(session.decrypt(x.inner_product(y))[5], u64(dot % t));
    EXPECT_EQ(session.decrypt(x.inner_product(ys))[5], u64(dot % t));

    // The same as single rotations.
    std::vector<int> shifts = {1, -2, 7, 0};
    auto hoisted = x.rotate_hoisted(shifts);
    ASSERT_EQ(hoisted.size(), shifts.size());
    for (std::size_t i = 0; i < shifts.size(); i++)
    {
        EXPECT_EQ(session.decrypt(hoisted[i]),
                  session.decrypt(x.rotate(shifts[i])))
            << "shift " << shifts[i];
    }

    std::vector<u64> a(n), b(n);
    for (int i = 0; i < n; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
    }
    auto ea = session.encrypt(a);
    auto eb = session.encrypt(b);
    auto gate = [&](const std::function<u64(u64, u64)>& f)
    {
        std::vector<u64> result(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = f(a[i], b[i]);
        }
        return result;
    };
    EXPECT_EQ(session.decrypt(heongpu::NOT(ea)),
              gate([](u64 p, u64) { return 1 - p; }));
    EXPECT_EQ(session.decrypt(heongpu::AND(ea, eb)),
              gate([](u64 p, u64 q) { return p & q; }));
    EXPECT_EQ(session.decrypt(heongpu::OR(ea, eb)),
              gate([](u64 p, u64 q) { return p | q; }));
    EXPECT_EQ(session.decrypt(heongpu::XOR(ea, eb)),
              gate([](u64 p, u64 q) { return p ^ q; }));
    EXPECT_EQ(session.decrypt(heongpu::NAND(ea, eb)),
              gate([](u64 p, u64 q) { return 1 - (p & q); }));
    EXPECT_EQ(session.decrypt(heongpu::NOR(ea, eb)),
              gate([](u64 p, u64 q) { return 1 - (p | q); }));
    EXPECT_EQ(session.decrypt(heongpu::XNOR(ea, eb)),
              gate([](u64 p, u64 q) { return 1 - (p ^ q); }));
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
