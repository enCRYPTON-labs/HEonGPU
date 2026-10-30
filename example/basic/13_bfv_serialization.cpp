// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <filesystem>

// These examples have been developed with reference to the Microsoft SEAL
// library.

// -----------------------------------------------------------------------------
// This example demonstrates how to serialize and deserialize various HEONGPU
// objects in a homomorphic‐encryption workflow. Serialization converts an
// in‐memory object (context, keys, plaintexts, ciphertexts, etc.) into a
// contiguous byte stream, which can then be stored or transmitted (e.g., over
// a network or to disk). Deserialization reverses the process, reconstructing
// the original object from its byte representation.
//
// Why use serialization?
//   • Portability: Send objects between client and server without sharing code.
//   • Persistence: Save intermediate results or parameters to disk and reload
//   them. • Interoperability: Exchange HE data with other languages or
//   services.
//
// Compression with Zlib (default serializer):
//   • Our serializer wraps each object’s save/load and then applies Zlib
//     compression on serialize(), and Zlib decompression on deserialize().
//   • Homomorphic objects contain large arrays of 64‐bit words (often with
//     many leading zeros), so compression can reduce size by 50–60%.
//   • Reduces bandwidth and storage costs.
//
// Optional raw‐binary serialization:
//   • You are not forced to use Zlib. Every HEONGPU object also supports
//     direct save(std::ostream&) / load(std::istream&) calls to write/read
//     raw binary “.bin” files with no compression.
//   • To bypass compression entirely, simply call object.save(your_ofs) and
//     object.load(your_ifs) on std::ofstream/ifstream opened with ios::binary.
//
// Convenient file‐I/O helpers:
//   • heongpu::serializer::save_to_file(obj, path) → serializes, compresses,
//     and writes to “path” in one call.
//   • heongpu::serializer::load_from_file<T>(path) → reads, decompresses, and
//     reconstructs the object of type T.
//
// Which objects can be serialized?
//   • HEContext      – encryption parameters, modulus settings, noise
//   schedules. • Secretkey      – secret key material for decryption. •
//   Publickey      – public key for encryption. • Relinkey       –
//   relinearization key for ciphertext multiplication. • Galoiskey      –
//   Galois key for ciphertext rotations. • Plaintext      – encoded vectors of
//   data in polynomial form. • Ciphertext     – encrypted data under the
//   scheme.
// -----------------------------------------------------------------------------

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::BFV;

int main(int argc, char* argv[])
{
    // 2. Set up HE context (BFV scheme)
    heongpu::HEParams<Scheme> params;
    const size_t poly_modulus_degree = 4096;
    params.poly_modulus_degree = poly_modulus_degree;
    params.plain_modulus = 1032193;
    heongpu::HEContext<Scheme> context = params.make_context();

    // 3. Serialize / deserialize the context
    std::stringstream ctx_stream;
    context->save(ctx_stream);
    heongpu::HEContext<Scheme> loaded_context = heongpu::GenHEContext<Scheme>();
    loaded_context->load(ctx_stream);
    loaded_context->print_parameters();

    // 4. Generate secret key
    heongpu::HEKeyGenerator<Scheme> keygen(loaded_context);
    heongpu::Secretkey<Scheme> secret_key(loaded_context);
    keygen.generate_secret_key(secret_key);

    // 5. Serialize / deserialize secret key via our serializer
    //    (serialization compresses the data using Zlib,
    //     deserialization decompresses it)
    auto sk_buffer = heongpu::serializer::serialize(secret_key);
    auto secret_key2 =
        heongpu::serializer::deserialize<heongpu::Secretkey<Scheme>>(sk_buffer);

    // 6. Generate and serialize public key
    heongpu::Publickey<Scheme> public_key(loaded_context);
    keygen.generate_public_key(public_key, secret_key2);
    heongpu::serializer::save_to_file(public_key, "public_key.bin");
    auto public_key2 =
        heongpu::serializer::load_from_file<heongpu::Publickey<Scheme>>(
            "public_key.bin");
    std::filesystem::remove("public_key.bin");

    // 7. Generate and serialize relinearization key
    heongpu::Relinkey<Scheme> relin_key(loaded_context);
    keygen.generate_relin_key(relin_key, secret_key2);
    std::stringstream rk_stream;
    relin_key.save(rk_stream);
    heongpu::Relinkey<Scheme> relin_key2;
    relin_key2.load(rk_stream);

    // 8. Generate and serialize Galois key (for rotations)
    std::vector<int> galois_shifts = {1, 3};
    heongpu::Galoiskey<Scheme> galois_key(loaded_context, galois_shifts);
    keygen.generate_galois_key(galois_key, secret_key2);
    std::stringstream gk_stream;
    galois_key.save(gk_stream);
    heongpu::Galoiskey<Scheme> galois_key2;
    galois_key2.load(gk_stream);

    // 9. Prepare encoder, encryptor, decryptor and HE operator
    heongpu::HEEncoder<Scheme> encoder(loaded_context);
    heongpu::HEEncryptor<Scheme> encryptor(loaded_context, public_key2);
    heongpu::HEDecryptor<Scheme> decryptor(loaded_context, secret_key2);
    heongpu::HEArithmeticOperator<Scheme> operators(loaded_context, encoder);

    // 10. Create and serialize plaintext
    const int row_size = int(poly_modulus_degree / 2);
    std::vector<uint64_t> message(poly_modulus_degree, 8ULL);
    // Fill in a few example slots
    message[0] = 1ULL;
    message[1] = 12ULL;
    message[2] = 23ULL;
    message[3] = 31ULL;
    message[row_size] = 7ULL;
    message[row_size + 1] = 54ULL;
    message[row_size + 2] = 6ULL;
    message[row_size + 3] = 100ULL;

    heongpu::Plaintext<Scheme> plain1(loaded_context);
    encoder.encode(plain1, message);
    std::stringstream pt_stream;
    plain1.save(pt_stream);
    heongpu::Plaintext<Scheme> plain2;
    plain2.load(pt_stream);

    // 11. Encrypt and serialize ciphertext
    heongpu::Ciphertext<Scheme> cipher1(loaded_context);
    encryptor.encrypt(cipher1, plain2);
    std::stringstream ct_stream;
    cipher1.save(ct_stream);
    heongpu::Ciphertext<Scheme> cipher2;
    cipher2.load(ct_stream);

    // 12. Perform homomorphic operations
    operators.multiply_inplace(cipher2, cipher2);
    operators.relinearize_inplace(cipher2, relin_key2);
    operators.rotate_rows_inplace(cipher2, galois_key2, 3);

    // 13. Decrypt and display result
    heongpu::Plaintext<Scheme> plain_result(loaded_context);
    decryptor.decrypt(plain_result, cipher2);
    std::vector<uint64_t> result;
    encoder.decode(result, plain_result);

    // [961, 64, 64, 64, 64, ..., 64, 64,  1,144,529 ]
    // [10000, 64, 64, 64, 64, ..., 64, 64, 49,2916, 36 ]

    std::cout << "Decrypted result matrix:\n";
    display_matrix(result, row_size);

    return EXIT_SUCCESS;
}
