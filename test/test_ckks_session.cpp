// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// HESession<Scheme::CKKS> and the operators of SessionCiphertext.

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Mode = heongpu::rescale_mode;
    using Session = heongpu::HESession<S>;
    using SessionCiphertext = heongpu::SessionCiphertext<S>;

    heongpu::HEParams<S> params(int depth)
    {
        heongpu::HEParams<S> p;
        p.multiplicative_depth = depth;
        return p;
    }

    std::vector<double> values(int count, double offset)
    {
        std::vector<double> result(count);
        for (int i = 0; i < count; i++)
        {
            result[i] = 0.5 * std::sin(0.01 * i + offset);
        }
        return result;
    }

    // The largest difference between the slots of result and expected.
    double max_error(const std::vector<double>& result,
                     const std::vector<double>& expected)
    {
        double error = 0.0;
        for (std::size_t i = 0; i < expected.size(); i++)
        {
            error = std::max(error, std::abs(result[i] - expected[i]));
        }
        return error;
    }

    std::vector<double> slotwise(const std::vector<double>& x,
                                 const std::function<double(double)>& function)
    {
        std::vector<double> result(x.size());
        std::transform(x.begin(), x.end(), result.begin(), function);
        return result;
    }

    std::vector<double>
    slotwise(const std::vector<double>& x, const std::vector<double>& y,
             const std::function<double(double, double)>& function)
    {
        std::vector<double> result(x.size());
        std::transform(x.begin(), x.end(), y.begin(), result.begin(), function);
        return result;
    }

    constexpr double tolerance = 1e-5;
} // namespace

TEST(CKKSSession, Rescale_Mode)
{
    Session automatic(params(2));
    EXPECT_EQ(automatic.get_rescale_mode(), Mode::automatic);
    EXPECT_EQ(automatic.scale(), std::pow(2.0, 40));
    EXPECT_EQ(automatic.slot_count(),
              int(automatic.context()->get_poly_modulus_degree() / 2));

    auto p = params(2);
    p.rescale_mode = Mode::manual;
    Session manual(p);
    EXPECT_EQ(manual.get_rescale_mode(), Mode::manual);

    // A parameter set keeps its mode, manual by default.
    Session from_set(params(2).generate());
    EXPECT_EQ(from_set.get_rescale_mode(), Mode::manual);
}

TEST(CKKSSession, Circuit)
{
    Session session(params(3));
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0), ys = values(n, 1.0);

    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    auto z = x * x * y + 0.5 * x - y;
    std::vector<double> result = session.decrypt(z);

    auto expected = slotwise(
        xs, ys, [](double a, double b) { return a * a * b + 0.5 * a - b; });
    EXPECT_LT(max_error(result, expected), tolerance);
}

TEST(CKKSSession, Operators)
{
    Session session(params(3));
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0), ys = values(n, 1.0);
    std::vector<double> vs = values(n, 2.0);
    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);

    using F = std::function<double(double, double)>;
    auto check = [&](const heongpu::SessionCiphertext<S>& c, const F& f) {
        EXPECT_LT(max_error(session.decrypt(c), slotwise(xs, ys, f)),
                  tolerance);
    };

    check(x + y, [](double a, double b) { return a + b; });
    check(x - y, [](double a, double b) { return a - b; });
    check(x * y, [](double a, double b) { return a * b; });
    check(-x, [](double a, double) { return -a; });
    check(x + 2.0, [](double a, double) { return a + 2.0; });
    check(2.0 + x, [](double a, double) { return 2.0 + a; });
    check(x - 2.0, [](double a, double) { return a - 2.0; });
    check(2.0 - x, [](double a, double) { return 2.0 - a; });
    check(x * 0.25, [](double a, double) { return a * 0.25; });
    check(-3.0 * x, [](double a, double) { return -3.0 * a; });

    // Vectors, with the values of vs.
    auto with_v = [&](const heongpu::SessionCiphertext<S>& c,
                      const std::function<double(double, double)>& f)
    {
        std::vector<double> expected(n);
        for (int i = 0; i < n; i++)
        {
            expected[i] = f(xs[i], vs[i]);
        }
        EXPECT_LT(max_error(session.decrypt(c), expected), tolerance);
    };
    with_v(x + vs, [](double a, double v) { return a + v; });
    with_v(vs + x, [](double a, double v) { return v + a; });
    with_v(x - vs, [](double a, double v) { return a - v; });
    with_v(vs - x, [](double a, double v) { return v - a; });
    with_v(x * vs, [](double a, double v) { return a * v; });
    with_v(vs * x, [](double a, double v) { return v * a; });
    // At a lower level, before the product is rescaled.
    with_v(x * x + vs, [](double a, double v) { return a * a + v; });
    with_v(x * x * vs, [](double a, double v) { return a * a * v; });

    // Compound assignments.
    auto z = x;
    z += y;
    z *= x;
    z -= 1.0;
    z *= 2.0;
    z += vs;
    z -= y;
    std::vector<double> expected(n);
    for (int j = 0; j < n; j++)
    {
        expected[j] = ((xs[j] + ys[j]) * xs[j] - 1.0) * 2.0 + vs[j] - ys[j];
    }
    EXPECT_LT(max_error(session.decrypt(z), expected), tolerance);
}

TEST(CKKSSession, Same_As_The_Lower_Level_API)
{
    Session session(params(3));
    int n = session.slot_count();
    auto x = session.encrypt(values(n, 0.0));
    auto y = session.encrypt(values(n, 1.0));
    auto z = x * x * y + x;

    auto& operators = session.operators();
    auto& X = x.ciphertext();
    auto& Y = y.ciphertext();
    auto X2 = operators.multiply(X, X);
    operators.relinearize_inplace(X2, session.relin_key());
    auto X2Y = operators.multiply(X2, Y);
    operators.relinearize_inplace(X2Y, session.relin_key());
    auto R = operators.add(X2Y, X);

    auto plain = session.decryptor().decrypt(R);
    EXPECT_EQ(session.decrypt(z), session.encoder().decode(plain));
    EXPECT_EQ(z.level(), R.level());
    EXPECT_EQ(z.scale(), R.scale());
}

TEST(CKKSSession, Rotations)
{
    Session session(params(2), {1, -2});
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0);
    auto x = session.encrypt(xs);

    auto rotated = [&](int shift)
    {
        std::vector<double> result(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = xs[(((i + shift) % n) + n) % n];
        }
        return result;
    };
    // Given up front, generated on first use, and their equivalents.
    for (int shift : {1, -2, 3, -5, n - 1, n, 0, 2 * n + 3})
    {
        EXPECT_LT(max_error(session.decrypt(x.rotate(shift)), rotated(shift)),
                  tolerance)
            << "shift " << shift;
    }
    // After a product, which the rotation rescales first.
    auto product = (x * x).rotate(3);
    EXPECT_LT(max_error(session.decrypt(product),
                        slotwise(rotated(3), [](double a) { return a * a; })),
              tolerance);

    // Conjugation.
    std::vector<Complex64> cs(n);
    for (int i = 0; i < n; i++)
    {
        cs[i] = Complex64(xs[i], 0.25 - xs[i]);
    }
    auto c = session.encrypt(cs).conjugate();
    auto result = session.decrypt_complex(c);
    double error = 0.0;
    for (int i = 0; i < n; i++)
    {
        error = std::max(error, std::abs(result[i].real() - cs[i].real()));
        error = std::max(error, std::abs(result[i].imag() + cs[i].imag()));
    }
    EXPECT_LT(error, tolerance);
}

TEST(CKKSSession, Conjugation_First)
{
    Session session(params(2));
    int n = session.slot_count();
    std::vector<Complex64> cs(n, Complex64(0.5, 0.25));
    auto c = session.encrypt(cs).conjugate();
    auto result = session.decrypt_complex(c);
    EXPECT_NEAR(result[7].imag(), -0.25, tolerance);
    // Then a rotation, added to the same key.
    std::vector<double> xs = values(n, 0.0);
    auto x = session.encrypt(xs).rotate(5);
    EXPECT_NEAR(session.decrypt(x)[0], xs[5], tolerance);
    session.add_rotations({7, 9});
    EXPECT_NEAR(session.decrypt(session.encrypt(xs).rotate(9))[0], xs[9],
                tolerance);
}

TEST(CKKSSession, Threads_Rotate_By_New_Steps)
{
    // Copies of a session share its Galois key; the threads add the keys of
    // their steps to it while the others rotate with it.
    Session session(params(1));
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0);
    auto x = session.encrypt(xs);

    constexpr int thread_count = 8;
    constexpr int rounds = 3;
    std::vector<std::string> failures(thread_count);
    std::vector<std::thread> threads;
    for (int t = 0; t < thread_count; t++)
    {
        threads.emplace_back(
            [&, t, copy = session]()
            {
                try
                {
                    for (int r = 0; r < rounds; r++)
                    {
                        int shift = 1 + t + thread_count * r;
                        std::vector<double> result =
                            copy.decrypt(x.rotate(shift));
                        for (int i = 0; i < n; i++)
                        {
                            if (std::abs(result[i] - xs[(i + shift) % n]) >
                                tolerance)
                            {
                                failures[t] = "wrong slot " +
                                              std::to_string(i) + " of shift " +
                                              std::to_string(shift);
                                return;
                            }
                        }
                    }
                }
                catch (const std::exception& error)
                {
                    failures[t] = error.what();
                }
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (int t = 0; t < thread_count; t++)
    {
        EXPECT_EQ(failures[t], "") << "thread " << t;
    }
}

TEST(CKKSSession, Sums_Hoisted_Rotations_And_Logic)
{
    Session session(params(3));
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0);
    std::vector<double> ys = values(n, 1.0);
    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);
    double sum = 0.0, dot = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += xs[i];
        dot += xs[i] * ys[i];
    }

    std::vector<double> sums = session.decrypt(x.sum_slots());
    EXPECT_NEAR(sums[0], sum, 1e-2);
    EXPECT_NEAR(sums[n - 1], sum, 1e-2);
    EXPECT_NEAR(session.decrypt(x.inner_product(y))[5], dot, 1e-2);
    EXPECT_NEAR(session.decrypt(x.inner_product(ys))[5], dot, 1e-2);

    // The same as single rotations.
    std::vector<int> shifts = {1, -2, 7, 0};
    auto hoisted = x.rotate_hoisted(shifts);
    ASSERT_EQ(hoisted.size(), shifts.size());
    for (std::size_t i = 0; i < shifts.size(); i++)
    {
        EXPECT_LT(max_error(session.decrypt(hoisted[i]),
                            session.decrypt(x.rotate(shifts[i]))),
                  tolerance)
            << "shift " << shifts[i];
    }

    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
    }
    auto ea = session.encrypt(a);
    auto eb = session.encrypt(b);
    auto gate = [&](const std::function<int(int, int)>& f)
    {
        std::vector<double> result(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = f(int(a[i]), int(b[i]));
        }
        return result;
    };
    EXPECT_LT(max_error(session.decrypt(heongpu::NOT(ea)),
                        gate([](int p, int) { return 1 - p; })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::AND(ea, eb)),
                        gate([](int p, int q) { return p & q; })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::OR(ea, eb)),
                        gate([](int p, int q) { return p | q; })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::XOR(ea, eb)),
                        gate([](int p, int q) { return p ^ q; })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::NAND(ea, eb)),
                        gate([](int p, int q) { return 1 - (p & q); })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::NOR(ea, eb)),
                        gate([](int p, int q) { return 1 - (p | q); })),
              tolerance);
    EXPECT_LT(max_error(session.decrypt(heongpu::XNOR(ea, eb)),
                        gate([](int p, int q) { return 1 - (p ^ q); })),
              tolerance);
    // One level for a gate of two inputs.
    EXPECT_EQ(heongpu::XOR(ea, eb).levels_left(), ea.levels_left() - 1);
    EXPECT_EQ(heongpu::XNOR(ea, eb).levels_left(), ea.levels_left() - 1);
}

TEST(CKKSSession, Polynomials)
{
    Session session(params(4));
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0);
    auto x = session.encrypt(xs);

    auto p = session.evaluate_polynomial(x, {0.5, -1.0, 0.25, 2.0});
    EXPECT_LT(max_error(session.decrypt(p), slotwise(xs,
                                                     [](double a) {
                                                         return 0.5 - a +
                                                                0.25 * a * a +
                                                                2.0 * a * a * a;
                                                     })),
              tolerance);

    auto e = session.evaluate_chebyshev(
        x, [](double a) { return std::exp(a); }, -1.0, 1.0, 7);
    EXPECT_LT(max_error(session.decrypt(e),
                        slotwise(xs, [](double a) { return std::exp(a); })),
              1e-4);

    // And more operations on the result.
    auto f = p * x + e;
    std::vector<double> expected(n);
    for (int i = 0; i < n; i++)
    {
        double a = xs[i];
        expected[i] =
            (0.5 - a + 0.25 * a * a + 2.0 * a * a * a) * a + std::exp(a);
    }
    EXPECT_LT(max_error(session.decrypt(f), expected), 1e-4);
}

TEST(CKKSSession, Manual_Rescale_Mode)
{
    auto p = params(3);
    p.rescale_mode = Mode::manual;
    Session session(p);
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0), ys = values(n, 1.0);
    std::vector<double> vs = values(n, 2.0);
    auto x = session.encrypt(xs);
    auto y = session.encrypt(ys);

    // Products are relinearized but not rescaled.
    auto z = x * y;
    EXPECT_TRUE(z.ciphertext().rescale_required());
    EXPECT_FALSE(z.ciphertext().relinearization_required());
    EXPECT_THROW(z * x, std::invalid_argument);

    session.operators().rescale_inplace(z.ciphertext());
    // A vector is added with the scale of the ciphertext and multiplied
    // with the scale of the parameters.
    auto w = z + vs;
    auto u = z * vs;
    session.operators().rescale_inplace(u.ciphertext());
    auto t = z * 0.5;
    std::vector<double> sum(n), product(n), half(n);
    for (int i = 0; i < n; i++)
    {
        sum[i] = xs[i] * ys[i] + vs[i];
        product[i] = xs[i] * ys[i] * vs[i];
        half[i] = xs[i] * ys[i] * 0.5;
    }
    EXPECT_LT(max_error(session.decrypt(w), sum), tolerance);
    EXPECT_LT(max_error(session.decrypt(u), product), tolerance);
    EXPECT_LT(max_error(session.decrypt(t), half), tolerance);
}

TEST(CKKSSession, Outlives_The_Session)
{
    std::optional<Session> session(std::in_place, params(2));
    int n = session->slot_count();
    std::vector<double> xs = values(n, 0.0);
    auto x = session->encrypt(xs);
    auto context = session->context();
    heongpu::Secretkey<S> secret_key = session->secret_key();
    session.reset();

    auto z = x * x + 1.0;
    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEDecryptor<S> decryptor(context, secret_key);
    auto plain = decryptor.decrypt(z.ciphertext());
    EXPECT_LT(max_error(encoder.decode(plain),
                        slotwise(xs, [](double a) { return a * a + 1.0; })),
              tolerance);
}

TEST(CKKSSession, Errors)
{
    Session first(params(2));
    Session second(params(2));
    std::vector<double> xs = values(first.slot_count(), 0.0);
    auto x = first.encrypt(xs);
    auto y = second.encrypt(xs);
    EXPECT_THROW(x + y, heongpu::ContextMismatchError);
    EXPECT_THROW(x * y, heongpu::ContextMismatchError);
    EXPECT_THROW(second.decrypt(x), heongpu::ContextMismatchError);
    // The context identity follows from the parameters.
    Session other(params(3));
    EXPECT_THROW(other.wrap(x.ciphertext()), heongpu::ContextMismatchError);

    // Copies share the session.
    Session copy = first;
    EXPECT_NO_THROW(copy.decrypt(x + first.wrap(x.ciphertext())));

    heongpu::SessionCiphertext<S> empty;
    EXPECT_THROW(x + empty, heongpu::StateError);
    EXPECT_THROW(empty * 2.0, heongpu::StateError);
    EXPECT_THROW(first.decrypt(empty), heongpu::StateError);
    empty = x;
    EXPECT_NO_THROW(first.decrypt(empty));
}

TEST(CKKSSession, Bootstrapping)
{
    // Regular bootstrapping with 4 levels between bootstrappings.
    Session session(heongpu::CKKSBootstrappingPreset::regular(4), {3});
    ASSERT_TRUE(session.has_bootstrapping());
    heongpu::BootstrappingLevels levels = session.bootstrapping_levels();
    EXPECT_EQ(levels.usable_levels(), 4);
    EXPECT_EQ(session.get_rescale_mode(), Mode::automatic);
    EXPECT_EQ(session.context()->get_top_level(), levels.output_level);

    int n = session.slot_count();
    std::vector<double> expected = values(n, 0.0);
    auto x = session.encrypt(expected);
    EXPECT_EQ(x.level(), levels.output_level);
    EXPECT_EQ(x.levels_left(), 4);

    int bootstraps = 0;
    for (int round = 0; round < 6; round++)
    {
        if (x.levels_left() < 2)
        {
            x = x.bootstrap();
            bootstraps++;
            EXPECT_EQ(x.level(), levels.output_level);
            EXPECT_EQ(x.levels_left(), 4);
        }
        x = 0.5 * x * x + 0.25;
        for (double& value : expected)
        {
            value = 0.5 * value * value + 0.25;
        }
    }
    EXPECT_EQ(bootstraps, 2);
    EXPECT_LT(max_error(session.decrypt(x), expected), std::pow(2.0, -10));

    // A step of the bootstrapping keys, a step given to the session and a
    // new one, added to the same key.
    for (int shift : {1, 3, 5})
    {
        std::vector<double> rotated = session.decrypt(x.rotate(shift));
        EXPECT_NEAR(rotated[0], expected[shift], std::pow(2.0, -10));
    }
}

TEST(CKKSSession, Slim_And_Bit_Bootstrapping)
{
    {
        Session session(heongpu::CKKSBootstrappingPreset::slim(2));
        int n = session.slot_count();
        std::vector<double> xs = values(n, 0.0);
        auto x = session.encrypt(xs);
        // The product waits for its rescaling.
        auto y = x * x;
        EXPECT_EQ(y.levels_left(), 1);
        auto b = y.bootstrap();
        EXPECT_EQ(b.levels_left(), 2);
        EXPECT_LT(max_error(session.decrypt(b),
                            slotwise(xs, [](double a) { return a * a; })),
                  std::pow(2.0, -15));

        // Below the input level of the bootstrapping.
        auto low = b * b * b * b;
        EXPECT_EQ(low.levels_left(), -1);
        EXPECT_THROW(low.bootstrap(), heongpu::LevelMismatchError);
    }
    {
        Session session(heongpu::CKKSBootstrappingPreset::bit(2));
        int n = session.slot_count();
        std::vector<double> bits(n);
        for (int i = 0; i < n; i++)
        {
            bits[i] = (i / 3) % 2;
        }
        auto x = session.encrypt(bits);
        // x * x is x for bits.
        auto b = (x * x).bootstrap();
        EXPECT_EQ(b.levels_left(), 2);
        EXPECT_LT(max_error(session.decrypt(b), bits), std::pow(2.0, -16));
        EXPECT_THROW(session.AND_bootstrapping(x, x), heongpu::StateError);
    }
}

TEST(CKKSSession, Gate_Bootstrapping)
{
    Session session(heongpu::CKKSBootstrappingPreset::gate(2));
    int n = session.slot_count();
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; i++)
    {
        a[i] = i % 2;
        b[i] = (i / 2) % 2;
    }
    auto x = session.encrypt(a);
    auto y = session.encrypt(b);

    struct Case
    {
        const char* name;
        std::function<SessionCiphertext(const Session&)> gate;
        std::function<int(int, int)> expected;
    };
    std::vector<Case> cases = {
        {"AND", [&](const Session& s) { return s.AND_bootstrapping(x, y); },
         [](int p, int q) { return p & q; }},
        {"OR", [&](const Session& s) { return s.OR_bootstrapping(x, y); },
         [](int p, int q) { return p | q; }},
        {"XOR", [&](const Session& s) { return s.XOR_bootstrapping(x, y); },
         [](int p, int q) { return p ^ q; }},
        {"NAND", [&](const Session& s) { return s.NAND_bootstrapping(x, y); },
         [](int p, int q) { return 1 - (p & q); }},
        {"NOR", [&](const Session& s) { return s.NOR_bootstrapping(x, y); },
         [](int p, int q) { return 1 - (p | q); }},
        {"XNOR", [&](const Session& s) { return s.XNOR_bootstrapping(x, y); },
         [](int p, int q) { return 1 - (p ^ q); }}};
    for (const Case& c : cases)
    {
        auto result = c.gate(session);
        EXPECT_EQ(result.levels_left(), 2) << c.name;
        std::vector<double> expected(n);
        for (int i = 0; i < n; i++)
        {
            expected[i] = c.expected(int(a[i]), int(b[i]));
        }
        EXPECT_LT(max_error(session.decrypt(result), expected),
                  std::pow(2.0, -16))
            << c.name;
    }
    EXPECT_THROW(x.bootstrap(), heongpu::StateError);

    // A session without bootstrapping.
    Session other(params(2));
    EXPECT_FALSE(other.has_bootstrapping());
    EXPECT_THROW(other.bootstrapping_levels(), heongpu::StateError);
    auto z = other.encrypt(values(other.slot_count(), 0.0));
    EXPECT_EQ(z.levels_left(), z.level());
    EXPECT_THROW(z.bootstrap(), heongpu::StateError);
    EXPECT_THROW(session.AND_bootstrapping(x, z),
                 heongpu::ContextMismatchError);
}

TEST(CKKSSession, Roles_Save_And_Load)
{
    using Role = heongpu::session_role;
    Session owner(params(2), {1});
    EXPECT_EQ(owner.role(), Role::owner);
    int n = owner.slot_count();
    std::vector<double> xs = values(n, 0.0);

    // A server gets the keys of the evaluator and a ciphertext.
    std::stringstream keys, input, output;
    owner.save(keys, Role::evaluator);
    owner.encrypt(xs).save(input);
    Session server = Session::load(keys, Role::evaluator);
    EXPECT_EQ(server.role(), Role::evaluator);
    EXPECT_EQ(server.get_rescale_mode(), Mode::automatic);
    auto x = server.load_ciphertext(input);
    ((x * x).rotate(1) + 0.5).save(output);
    EXPECT_THROW(server.decrypt(x), heongpu::StateError);
    EXPECT_THROW(server.secret_key(), heongpu::StateError);
    EXPECT_THROW(x.rotate(2), heongpu::KeyNotFoundError);
    EXPECT_THROW(server.add_rotations({2}), heongpu::KeyNotFoundError);

    std::vector<double> expected(n);
    for (int i = 0; i < n; i++)
    {
        expected[i] = xs[(i + 1) % n] * xs[(i + 1) % n] + 0.5;
    }
    EXPECT_LT(max_error(owner.decrypt(owner.load_ciphertext(output)), expected),
              tolerance);
    // Ciphertexts the server encrypts.
    std::stringstream encrypted;
    server.encrypt(xs).save(encrypted);
    EXPECT_LT(max_error(owner.decrypt(owner.load_ciphertext(encrypted)), xs),
              tolerance);

    // A client gets the public key.
    std::stringstream public_keys, sent;
    owner.save(public_keys, Role::encryptor);
    Session client = Session::load(public_keys);
    EXPECT_EQ(client.role(), Role::encryptor);
    auto c = client.encrypt(xs) + 1.0;
    EXPECT_THROW(c * c, heongpu::KeyNotFoundError);
    EXPECT_THROW(c.rotate(1), heongpu::KeyNotFoundError);
    EXPECT_THROW(client.relin_key(), heongpu::KeyNotFoundError);
    c.save(sent);
    EXPECT_LT(max_error(owner.decrypt(owner.load_ciphertext(sent)),
                        slotwise(xs, [](double a) { return a + 1.0; })),
              tolerance);

    // The owner with all its keys, which adds rotations again.
    std::stringstream all, again_input;
    owner.save(all, Role::owner);
    Session again = Session::load(all, Role::owner);
    EXPECT_EQ(again.role(), Role::owner);
    owner.encrypt(xs).save(again_input);
    auto z = again.load_ciphertext(again_input);
    EXPECT_NEAR(again.decrypt(z.rotate(5))[0], xs[5], tolerance);

    // Roles that do not match.
    std::stringstream evaluator_keys, owner_keys, sink;
    owner.save(evaluator_keys, Role::evaluator);
    owner.save(owner_keys, Role::owner);
    EXPECT_THROW(Session::load(evaluator_keys, Role::owner),
                 heongpu::StateError);
    EXPECT_THROW(Session::load(owner_keys, Role::evaluator),
                 heongpu::StateError);
    EXPECT_THROW(server.save(sink, Role::owner), heongpu::StateError);
    EXPECT_THROW(client.save(sink, Role::evaluator), heongpu::StateError);
    EXPECT_THROW(owner.save(sink, static_cast<Role>(7)),
                 heongpu::ParameterError);

    // A ciphertext of other parameters, and data that is not a session.
    Session other(params(3));
    std::stringstream foreign;
    other.encrypt(xs).save(foreign);
    EXPECT_THROW(owner.load_ciphertext(foreign), heongpu::ContextMismatchError);
    std::stringstream not_a_session;
    owner.encrypt(xs).save(not_a_session);
    EXPECT_THROW(Session::load(not_a_session), heongpu::SerializationError);
    std::stringstream short_data;
    owner.save(short_data, Role::evaluator);
    std::stringstream truncated(short_data.str().substr(0, 20));
    EXPECT_THROW(Session::load(truncated), heongpu::SerializationError);
    std::stringstream empty;
    EXPECT_THROW(Session::load(empty), heongpu::SerializationError);
}

TEST(CKKSSession, Saves_The_Manual_Rescale_Mode)
{
    heongpu::HEParams<S> p = params(2);
    p.rescale_mode = Mode::manual;
    Session owner(p);
    std::stringstream keys;
    owner.save(keys, heongpu::session_role::evaluator);
    Session server = Session::load(keys);
    EXPECT_EQ(server.get_rescale_mode(), Mode::manual);
    EXPECT_EQ(server.scale(), owner.scale());
}

TEST(CKKSSession, Bootstrapping_Evaluator)
{
    using Role = heongpu::session_role;
    Session owner(heongpu::CKKSBootstrappingPreset::regular(1));
    std::stringstream keys, input, output;
    owner.save(keys, Role::evaluator);
    Session server = Session::load(keys, Role::evaluator);
    ASSERT_TRUE(server.has_bootstrapping());
    EXPECT_EQ(server.bootstrapping_levels().output_level,
              owner.bootstrapping_levels().output_level);
    EXPECT_EQ(server.context()->get_top_level(),
              owner.context()->get_top_level());

    int n = owner.slot_count();
    std::vector<double> xs = values(n, 0.0);
    owner.encrypt(xs).save(input);
    auto x = server.load_ciphertext(input);
    auto y = (x * x).bootstrap();
    EXPECT_EQ(y.levels_left(), 1);
    y.save(output);
    EXPECT_LT(max_error(owner.decrypt(owner.load_ciphertext(output)),
                        slotwise(xs, [](double a) { return a * a; })),
              std::pow(2.0, -10));
}

TEST(CKKSSession, Streams)
{
    // Threads compute on their own non-blocking streams with views of one
    // session, on a ciphertext of the default stream, and add rotation keys.
    Session session(params(2), {1});
    int n = session.slot_count();
    std::vector<double> xs = values(n, 0.0);
    auto shared = session.encrypt(xs);
    std::vector<double> reference =
        session.decrypt((shared * shared).rotate(1) + shared);
    session.synchronize();

    constexpr int thread_count = 4;
    std::vector<cudaStream_t> streams(thread_count);
    for (cudaStream_t& stream : streams)
    {
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  cudaSuccess);
    }
    std::vector<std::string> failures(thread_count);
    std::vector<std::thread> threads;
    for (int t = 0; t < thread_count; t++)
    {
        threads.emplace_back(
            [&, t]()
            {
                try
                {
                    Session view = session.on_stream(streams[t]);
                    for (int round = 0; round < 3; round++)
                    {
                        auto x = view.encrypt(xs);
                        auto y = (x * shared).rotate(1) + x;
                        if (y.stream() != streams[t])
                        {
                            failures[t] = "result on another stream";
                            return;
                        }
                        if (max_error(view.decrypt(y), reference) > tolerance)
                        {
                            failures[t] = "wrong result";
                            return;
                        }
                        int shift = 2 + t + thread_count * round;
                        std::vector<double> rotated =
                            view.decrypt(x.rotate(shift));
                        if (std::abs(rotated[0] - xs[shift]) > tolerance)
                        {
                            failures[t] = "wrong rotation";
                            return;
                        }
                    }
                }
                catch (const std::exception& error)
                {
                    failures[t] = error.what();
                }
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (int t = 0; t < thread_count; t++)
    {
        EXPECT_EQ(failures[t], "") << "thread " << t;
        EXPECT_EQ(cudaStreamDestroy(streams[t]), cudaSuccess);
    }
    EXPECT_EQ(session.stream(), nullptr);
}
