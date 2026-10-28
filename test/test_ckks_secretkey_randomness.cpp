// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <sstream>

#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
#include <dlfcn.h>

// Replaces the OpenSSL function for HEonGPU, whether HEonGPU is linked
// statically or as a shared library, and forwards to the OpenSSL one.
namespace
{
    using RandomFunction = int (*)(unsigned char*, int);
    RandomFunction openssl_function(const char* name)
    {
        return reinterpret_cast<RandomFunction>(dlsym(RTLD_NEXT, name));
    }
} // namespace

namespace
{
    int private_random_calls_until_failure = -1;
}

extern "C" int RAND_priv_bytes(unsigned char* buffer, int size)
{
    if (private_random_calls_until_failure == 0)
    {
        return 0;
    }
    if (private_random_calls_until_failure > 0)
    {
        --private_random_calls_until_failure;
    }
    static const RandomFunction real = openssl_function("RAND_priv_bytes");
    return real(buffer, size);
}
#endif

namespace
{
    using heongpu::CudaException;
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;

    heongpu::HEContext<scheme> make_context()
    {
        auto context = heongpu::GenHEContext<scheme>();
        context->set_poly_modulus_degree(degree);
        context->set_coeff_modulus_bit_sizes({30, 30}, {30});
        heongpu::MemoryPoolConfig config;
        config.use_memory_pool = false;
        context->generate(config);
        return context;
    }

    std::vector<Data64> read_key(heongpu::Secretkey<scheme>& key)
    {
        std::vector<Data64> result(degree * key.coeff_modulus_count());
        HEONGPU_CUDA_CHECK(cudaMemcpy(result.data(), key.data(),
                                      result.size() * sizeof(Data64),
                                      cudaMemcpyDeviceToHost));
        return result;
    }
} // namespace

TEST(HEonGPU, CKKS_SecretKeysDoNotRepeatWithLegacySeed)
{
    auto context = make_context();
    heongpu::HEKeyGenerator<scheme> keygen(context);
    heongpu::HEKeyGenerator<scheme> other_keygen(context);
    for (int weight : {32, 192, degree / 2, degree})
    {
        SCOPED_TRACE(weight);
        heongpu::Secretkey<scheme> first(context, weight);
        heongpu::Secretkey<scheme> second(context, weight);
        heongpu::Secretkey<scheme> third(context, weight);
        // set_seed, deprecated, has no effect.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        keygen.set_seed(12345);
        keygen.generate_secret_key(first);
        keygen.set_seed(12345);
        keygen.generate_secret_key(second);
        other_keygen.set_seed(12345);
        other_keygen.generate_secret_key(third);
#pragma GCC diagnostic pop
        auto first_data = read_key(first);
        EXPECT_NE(first_data, read_key(second));
        EXPECT_NE(first_data, read_key(third));
    }
}

TEST(HEonGPU, CKKS_SecretKeysKeepExactHammingWeightAndTernaryCoefficients)
{
    auto context = make_context();
    const auto moduli = context->get_key_modulus();
    const auto roots =
        heongpu::generate_primitive_root_of_unity(degree, moduli);
    const auto modulus = moduli[0];
    const auto psi = roots[0];
    gpuntt::NTTFactors<Data64> factors(
        modulus, OPERATOR64::mult(psi, psi, modulus), psi);
    gpuntt::NTTCPU<Data64> inverse(gpuntt::NTTParameters<Data64>(
        context->get_log_poly_modulus_degree(), factors, gpuntt::X_N_plus));
    heongpu::HEKeyGenerator<scheme> keygen(context);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int weight : {1, 32, 192, degree / 2, degree})
    {
        SCOPED_TRACE(weight);
        heongpu::Secretkey<scheme> key(context, weight);
        keygen.generate_secret_key(
            key, heongpu::ExecutionOptions().set_stream(stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto ntt_data = read_key(key);
        ntt_data.resize(degree);
        auto coefficients = inverse.intt(ntt_data);
        int nonzero = 0;
        for (auto coefficient : coefficients)
        {
            EXPECT_TRUE(coefficient == 0 || coefficient == 1 ||
                        coefficient == modulus.value - 1);
            nonzero += coefficient != 0;
        }
        EXPECT_EQ(nonzero, weight);
    }
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(HEonGPU, CKKS_CryptographicSparseKeysEncryptAndDecrypt)
{
    auto context = make_context();
    heongpu::HEKeyGenerator<scheme> keygen(context);
    heongpu::HEEncoder<scheme> encoder(context);
    const std::vector<double> message(degree / 2, 0.125);
    for (int weight : {32, 192})
    {
        SCOPED_TRACE(weight);
        heongpu::Secretkey<scheme> secret(context, weight);
        keygen.generate_secret_key(secret);
        heongpu::Publickey<scheme> public_key(context);
        keygen.generate_public_key(public_key, secret);
        heongpu::HEEncryptor<scheme> encryptor(context, public_key);
        heongpu::HEDecryptor<scheme> decryptor(context, secret);
        heongpu::Plaintext<scheme> plaintext(context), decrypted(context);
        heongpu::Ciphertext<scheme> ciphertext(context);
        encoder.encode(plaintext, message, std::pow(2.0, 25));
        encryptor.encrypt(ciphertext, plaintext);
        decryptor.decrypt(decrypted, ciphertext);
        std::vector<double> result;
        encoder.decode(result, decrypted);
        ASSERT_EQ(result.size(), message.size());
        for (size_t i = 0; i < result.size(); ++i)
        {
            EXPECT_NEAR(result[i], message[i], 1e-4);
        }
    }
}

#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
TEST(HEonGPU, CKKS_SecretKeyRandomFailureDoesNotGenerateAKey)
{
    auto context = make_context();
    heongpu::HEKeyGenerator<scheme> keygen(context);
    // Exercise failure on the first request and after a successful refill.
    for (int successful_calls : {0, 1})
    {
        SCOPED_TRACE(successful_calls);
        heongpu::Secretkey<scheme> key(context, degree / 2);
        private_random_calls_until_failure = successful_calls;
        EXPECT_THROW(keygen.generate_secret_key(key), std::runtime_error);
        private_random_calls_until_failure = -1;
        std::stringstream output;
        EXPECT_THROW(key.save(output), std::runtime_error);
        EXPECT_NO_THROW(keygen.generate_secret_key(key));
        EXPECT_NO_THROW(key.save(output));
    }
}
#endif
