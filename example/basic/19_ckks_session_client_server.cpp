// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <cmath>
#include <sstream>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;
using Session = heongpu::HESession<Scheme>;

int main(int argc, char* argv[])
{
    // The client owns the keys. It sends the server the keys of the
    // evaluator: the public key and the evaluation keys, without the secret
    // key. String streams stand for the network or files here.
    heongpu::HEParams<Scheme> params;
    params.multiplicative_depth = 2;
    Session client(params, {1});
    std::stringstream keys;
    client.save(keys, heongpu::session_role::evaluator);
    std::cout << "Keys of the evaluator: " << keys.str().size() << " bytes"
              << std::endl;

    // The server loads them and checks the role, so that it never holds a
    // secret key it was sent by mistake.
    Session server = Session::load(keys, heongpu::session_role::evaluator);

    const int slot_count = client.slot_count();
    std::vector<double> xs(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        xs[i] = 0.5 * std::sin(0.01 * i);
    }
    std::stringstream request;
    client.encrypt(xs).save(request);

    // The server computes x * x + x rotated by one slot. It cannot decrypt.
    auto x = server.load_ciphertext(request);
    auto y = (x * x + x).rotate(1);
    std::stringstream response;
    y.save(response);
    try
    {
        server.decrypt(y);
    }
    catch (const heongpu::StateError& error)
    {
        std::cout << "The server cannot decrypt: " << error.what() << std::endl;
    }

    // The client decrypts the result.
    std::vector<double> result =
        client.decrypt(client.load_ciphertext(response));
    std::vector<double> expected(slot_count);
    double error = 0.0;
    for (int i = 0; i < slot_count; i++)
    {
        double value = xs[(i + 1) % slot_count];
        expected[i] = value * value + value;
        error = std::max(error, std::abs(result[i] - expected[i]));
    }
    std::cout << "Expected:" << std::endl;
    display_vector(expected);
    std::cout << "Result:" << std::endl;
    display_vector(result);
    std::cout << "Maximum error: " << error << std::endl;

    // A third party with the public key alone encrypts data for the server.
    std::stringstream public_key;
    client.save(public_key, heongpu::session_role::encryptor);
    Session sensor =
        Session::load(public_key, heongpu::session_role::encryptor);
    std::stringstream data, sum;
    sensor.encrypt(std::vector<double>(slot_count, 0.25)).save(data);
    (server.load_ciphertext(data) + x).save(sum);
    std::cout << "Slot 0 of the sum: "
              << client.decrypt(client.load_ciphertext(sum))[0] << ", expected "
              << 0.25 + xs[0] << std::endl;

    return EXIT_SUCCESS;
}
