// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>

#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
namespace
{
    int calls_until_failure = -1;
    int random_calls = 0;
} // namespace
extern "C" int __real_RAND_priv_bytes(unsigned char*, int);
extern "C" int __wrap_RAND_priv_bytes(unsigned char* data, int size)
{
    ++random_calls;
    if (calls_until_failure == 0)
        return 0;
    if (calls_until_failure > 0)
        --calls_until_failure;
    return __real_RAND_priv_bytes(data, size);
}
struct RandomFailure
{
    explicit RandomFailure(int calls) { calls_until_failure = calls; }
    ~RandomFailure() { calls_until_failure = -1; }
};
#endif

namespace
{
    using heongpu::CudaException;
    constexpr int degree = 4096;
    template <heongpu::Scheme S>
    heongpu::HEContext<S> make_context(int n = degree, int bits = 30)
    {
        auto context = heongpu::GenHEContext<S>();
        context->set_poly_modulus_degree(n);
        context->set_coeff_modulus_bit_sizes({bits, bits}, {bits});
        if constexpr (S == heongpu::Scheme::BFV)
            context->set_plain_modulus(65537);
        heongpu::MemoryPoolConfig config;
        config.use_memory_pool = false;
        context->generate(config);
        return context;
    }
    template <heongpu::Scheme S> std::string save(heongpu::Secretkey<S>& key)
    {
        std::stringstream stream;
        // The checks below parse the field layout of the format v1.
        key.save(stream, heongpu::serialization_format::v1);
        return stream.str();
    }
    template <typename T> std::vector<T> read(T* data, size_t size, bool device)
    {
        std::vector<T> result(size);
        if (device)
        {
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
            HEONGPU_CUDA_CHECK(cudaMemcpy(result.data(), data, size * sizeof(T),
                                          cudaMemcpyDeviceToHost));
        }
        else
            std::copy_n(data, size, result.data());
        return result;
    }
    auto tfhe_context()
    {
        heongpu::MemoryPoolConfig config;
        config.use_memory_pool = false;
        return heongpu::GenHEContext<heongpu::Scheme::TFHE>(config);
    }
} // namespace

template <typename T> class RNSSecretKey : public ::testing::Test
{
};
using RNSSchemes = ::testing::Types<
    std::integral_constant<heongpu::Scheme, heongpu::Scheme::BFV>,
    std::integral_constant<heongpu::Scheme, heongpu::Scheme::CKKS>>;
TYPED_TEST_SUITE(RNSSecretKey, RNSSchemes);

TYPED_TEST(RNSSecretKey, PrivateRandomnessExactWeightCopyAndSerialization)
{
    constexpr auto S = TypeParam::value;
    auto context = make_context<S>();
    heongpu::HEKeyGenerator<S> generator(context), other_generator(context);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int weight : {1, 32, 192, degree / 2, degree})
    {
        SCOPED_TRACE(weight);
        heongpu::Secretkey<S> ungenerated(context, weight);
        heongpu::Secretkey<S> first(ungenerated), second(ungenerated);
        generator.set_seed(12345);
        generator.generate_secret_key(
            first, heongpu::ExecutionOptions().set_stream(stream));
        generator.set_seed(12345);
        generator.generate_secret_key(
            second, heongpu::ExecutionOptions().set_stream(stream));
        auto bytes = save(first);
        if (weight != 1)
            EXPECT_NE(bytes, save(second));
        heongpu::Secretkey<S> third(ungenerated);
        other_generator.set_seed(12345);
        other_generator.generate_secret_key(third);
        if (weight != 1)
            EXPECT_NE(bytes, save(third));
        // Loading verifies every RNS limb is ternary with the declared exact
        // weight.
        heongpu::Secretkey<S> restored(context);
        std::stringstream input(bytes);
        restored.load(input);
        EXPECT_EQ(bytes, save(restored));
        first.store_in_host(stream);
        heongpu::Secretkey<S> host_copy(first);
        host_copy.store_in_device(stream);
        EXPECT_EQ(save(host_copy), bytes);
        auto decoded = heongpu::serializer::deserialize<heongpu::Secretkey<S>>(
            heongpu::serializer::serialize(second));
        // The legacy default-constructed deserialize path binds when used.
        heongpu::HEDecryptor<S> decryptor(context, decoded);
        EXPECT_EQ(save(decoded), save(second));
    }
    heongpu::Secretkey<S> host_key(context, 192);
    generator.generate_secret_key(
        host_key,
        heongpu::ExecutionOptions().set_stream(stream).set_storage_type(
            heongpu::storage_type::HOST));
    EXPECT_FALSE(host_key.is_on_device());
    heongpu::Secretkey<S> loaded_host(context);
    std::stringstream host_input(save(host_key));
    EXPECT_NO_THROW(loaded_host.load(host_input));
    EXPECT_EQ(
        read(host_key.data(), degree * host_key.coeff_modulus_count(), false),
        read(loaded_host.data(), degree * loaded_host.coeff_modulus_count(),
             true));
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}

TYPED_TEST(RNSSecretKey, RejectsTruncationForgedMetadataAndInvalidPolynomial)
{
    constexpr auto S = TypeParam::value;
    auto context = make_context<S>();
    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> source(context, 192);
    generator.generate_secret_key(source);
    const auto bytes = save(source);
    std::vector<std::string> invalid{bytes.substr(0, 1),
                                     bytes.substr(0, bytes.size() - 1)};
    auto invalid_bool = bytes;
    invalid_bool[sizeof(heongpu::scheme_type) + 4 * sizeof(int)] = 2;
    invalid.push_back(invalid_bool);
    auto invalid_size = bytes;
    uint32_t huge_size = UINT32_MAX;
    const size_t payload_offset =
        sizeof(heongpu::scheme_type) + 4 * sizeof(int) + 2 * sizeof(bool) +
        sizeof(heongpu::storage_type) + sizeof(uint32_t);
    std::memcpy(invalid_size.data() + payload_offset - sizeof(uint32_t),
                &huge_size, sizeof(huge_size));
    invalid.push_back(invalid_size);
    auto zero_key = bytes;
    std::fill(zero_key.begin() + payload_offset, zero_key.end(), 0);
    invalid.push_back(zero_key);
    for (auto& bad : invalid)
    {
        heongpu::Secretkey<S> key(context);
        std::stringstream input(bad);
        EXPECT_THROW(key.load(input), std::exception);
        EXPECT_THROW(save(key), std::runtime_error);
        EXPECT_NO_THROW(generator.generate_secret_key(key));
    }
    auto larger = make_context<S>(8192);
    heongpu::Secretkey<S> wrong_context(larger);
    std::stringstream input(bytes);
    EXPECT_THROW(wrong_context.load(input), std::invalid_argument);
    EXPECT_THROW(generator.generate_secret_key(wrong_context),
                 std::invalid_argument);
    EXPECT_THROW((heongpu::HEDecryptor<S>(larger, source)),
                 std::invalid_argument);
    EXPECT_THROW(source.set_context(larger), std::invalid_argument);
    auto different_primes = make_context<S>(degree, 31);
    heongpu::Secretkey<S> other(different_primes);
    std::stringstream different_input(bytes);
    EXPECT_THROW(other.load(different_input), std::exception);
    EXPECT_THROW((heongpu::HEDecryptor<S>(different_primes, source)),
                 std::invalid_argument);
}

TYPED_TEST(RNSSecretKey,
           RawKeysRequireTernaryCoefficientsAndPreserveTheirWeight)
{
    constexpr auto S = TypeParam::value;
    auto context = make_context<S>();
    std::vector<int> values(degree, 0);
    EXPECT_THROW((heongpu::Secretkey<S>(values, context)),
                 std::invalid_argument);
    values[0] = 1;
    values[7] = -1;
    values[192] = 1;
    heongpu::Secretkey<S> key(values, context);
    std::stringstream input(save(key));
    heongpu::Secretkey<S> restored(context);
    EXPECT_NO_THROW(restored.load(input));
    values[123] = 2;
    EXPECT_THROW((heongpu::Secretkey<S>(values, context)),
                 std::invalid_argument);
    heongpu::HostVector<int> pinned(values.size());
    std::copy(values.begin(), values.end(), pinned.begin());
    EXPECT_THROW((heongpu::Secretkey<S>(pinned, context)),
                 std::invalid_argument);
    heongpu::Secretkey<S> empty(context);
    EXPECT_THROW((heongpu::HEDecryptor<S>(context, empty)), std::logic_error);
    heongpu::Secretkey<S> moved(std::move(key));
    EXPECT_THROW(save(key), std::runtime_error);
    EXPECT_NO_THROW(save(moved));
    empty.set_context(context);
    heongpu::HEKeyGenerator<S> generator(context);
    EXPECT_NO_THROW(generator.generate_secret_key(empty));
}

#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
TYPED_TEST(RNSSecretKey, RandomFailureLeavesNoUsableKeyAndAllowsRetry)
{
    constexpr auto S = TypeParam::value;
    auto context = make_context<S>();
    heongpu::HEKeyGenerator<S> generator(context);
    for (int successes : {0, 1})
    {
        heongpu::Secretkey<S> key(context);
        {
            RandomFailure failure(successes);
            EXPECT_THROW(generator.generate_secret_key(key),
                         std::runtime_error);
        }
        EXPECT_THROW(save(key), std::runtime_error);
        EXPECT_NO_THROW(generator.generate_secret_key(key));
    }
}
#endif

TEST(SecretKeySecurity, TFHEIndependentCompleteBinaryKeysAndHostTransfers)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = tfhe_context();
    heongpu::HEKeyGenerator<S> generator(context);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    heongpu::Secretkey<S> first(context), second;
    second.set_context(context);
    generator.generate_secret_key(
        first, heongpu::ExecutionOptions().set_stream(stream).set_storage_type(
                   heongpu::storage_type::HOST));
    generator.generate_secret_key(
        second, heongpu::ExecutionOptions().set_stream(stream));
    EXPECT_FALSE(first.is_on_device());
    auto lwe = read(first.lwe_data(), first.lwe_size(), first.is_on_device());
    auto tlwe =
        read(first.tlwe_data(), first.tlwe_size(), first.is_on_device());
    ASSERT_EQ(lwe.size(), 512u);
    ASSERT_EQ(tlwe.size(), 1024u);
    for (int value : tlwe)
        EXPECT_TRUE(value == 0 || value == 1);
    for (int value : lwe)
        EXPECT_TRUE(value == 0 || value == 1);
    EXPECT_NE(lwe, std::vector<int32_t>(tlwe.begin(), tlwe.begin() + 512));
    EXPECT_NE(lwe, read(second.lwe_data(), second.lwe_size(), true));
    EXPECT_NE(tlwe, read(second.tlwe_data(), second.tlwe_size(), true));
    int tail_ones = std::count(tlwe.begin() + 512, tlwe.end(), 1);
    EXPECT_GT(tail_ones, 128);
    EXPECT_LT(tail_ones, 384);
    first.store_in_device(stream);
    first.store_in_host(stream);
    EXPECT_EQ(tlwe, read(first.tlwe_data(), first.tlwe_size(), false));
    heongpu::Secretkey<S> copied(first), moved(std::move(copied));
    EXPECT_FALSE(copied.is_generated());
    EXPECT_EQ(tlwe, read(moved.tlwe_data(), moved.tlwe_size(), false));
    second = moved;
    EXPECT_EQ(lwe, read(second.lwe_data(), second.lwe_size(), false));
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SecretKeySecurity,
     TFHEEncryptionHasNoiseAndWorksWithHostKeysAndNonblockingStream)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = tfhe_context();
    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> key(context);
    generator.generate_secret_key(key,
                                  heongpu::ExecutionOptions().set_storage_type(
                                      heongpu::storage_type::HOST));
    auto secret = read(key.lwe_data(), key.lwe_size(), false);
    heongpu::HEEncryptor<S> encryptor(context, key);
    heongpu::HEDecryptor<S> decryptor(context, key);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    heongpu::Ciphertext<S> cipher(context), another(context);
    std::vector<bool> bits(64);
    for (size_t i = 0; i < bits.size(); ++i)
        bits[i] = (i % 3 == 0);
    auto options = heongpu::ExecutionOptions().set_stream(stream);
    encryptor.encrypt(cipher, bits, options);
    std::vector<bool> decoded;
    decryptor.decrypt(cipher, decoded, options);
    EXPECT_EQ(decoded, bits);
    auto a = read(cipher.a_data(), bits.size() * key.lwe_size(), true);
    auto b = read(cipher.b_data(), bits.size(), true);
    int nonzero_noise = 0;
    double square_sum = 0;
    for (size_t row = 0; row < bits.size(); ++row)
    {
        uint32_t phase = uint32_t(b[row]);
        for (int i = 0; i < key.lwe_size(); ++i)
            phase -=
                uint32_t(a[row * key.lwe_size() + i]) * uint32_t(secret[i]);
        const int32_t message = bits[row] ? (1 << 29) : -(1 << 29);
        const int32_t noise = static_cast<int32_t>(phase - uint32_t(message));
        nonzero_noise += noise != 0;
        square_sum += double(noise) * double(noise);
    }
    EXPECT_GT(nonzero_noise, 60);
    const double rms = std::sqrt(square_sum / bits.size()) / 4294967296.0;
    EXPECT_GT(rms, 1e-5);
    EXPECT_LT(rms, 5e-5);
    encryptor.encrypt(another, bits, options);
    EXPECT_NE(a, read(another.a_data(), bits.size() * key.lwe_size(), true));
    std::vector<bool> empty;
    encryptor.encrypt(another, empty, options);
    decryptor.decrypt(another, decoded, options);
    EXPECT_TRUE(decoded.empty());
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}

#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
TEST(SecretKeySecurity, TFHERandomFailureRejectsKeysEncryptorsAndEvaluationKeys)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = tfhe_context();
    heongpu::HEKeyGenerator<S> generator(context);
    for (int successes : {0, 1, 5})
    {
        heongpu::Secretkey<S> key(context);
        {
            RandomFailure failure(successes);
            EXPECT_THROW(generator.generate_secret_key(key),
                         std::runtime_error);
        }
        EXPECT_FALSE(key.is_generated());
        EXPECT_THROW((heongpu::HEEncryptor<S>(context, key)),
                     std::runtime_error);
        EXPECT_THROW((heongpu::HEDecryptor<S>(context, key)), std::logic_error);
        EXPECT_NO_THROW(generator.generate_secret_key(key));
    }
    heongpu::Secretkey<S> key(context);
    generator.generate_secret_key(key);
    heongpu::Bootstrappingkey<S> boot(context);
    {
        RandomFailure failure(0);
        EXPECT_THROW(generator.generate_bootstrapping_key(boot, key),
                     std::runtime_error);
    }
    EXPECT_FALSE(boot.is_generated());
    heongpu::HEEncryptor<S> encryptor(context, key);
    heongpu::Ciphertext<S> cipher(context);
    const std::vector<bool> original{true, false, true, true};
    encryptor.encrypt(cipher, original);
    const auto original_a =
        read(cipher.a_data(), original.size() * key.lwe_size(), true);
    const auto original_b = read(cipher.b_data(), original.size(), true);
    for (int successes : {0, 1})
    {
        {
            RandomFailure failure(successes);
            EXPECT_THROW(encryptor.encrypt(cipher, {true, false}),
                         std::runtime_error);
        }
        EXPECT_EQ(cipher.size(), original.size());
        EXPECT_EQ(original_a, read(cipher.a_data(), original_a.size(), true));
        EXPECT_EQ(original_b, read(cipher.b_data(), original_b.size(), true));
    }
}
#endif

TEST(SecretKeySecurity,
     SerializerHandlesHighlyCompressibleDataAndRejectsTruncation)
{
    std::vector<uint8_t> raw(100000, 0);
    auto compressed = heongpu::serializer::compress(raw);
    ASSERT_LT(compressed.size() * 4, raw.size());
    EXPECT_EQ(heongpu::serializer::decompress(compressed), raw);
    compressed.pop_back();
    EXPECT_THROW(heongpu::serializer::decompress(compressed),
                 std::runtime_error);
}

TEST(SecretKeySecurity,
     TFHEEvaluationKeysSurviveHostTransferAndRefreshRandomness)
{
    constexpr auto S = heongpu::Scheme::TFHE;
    auto context = tfhe_context();
    heongpu::HEKeyGenerator<S> generator(context);
    heongpu::Secretkey<S> key(context);
    generator.generate_secret_key(key);
    heongpu::Bootstrappingkey<S> boot;
    boot.set_context(context);
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
    const int initial_calls = random_calls;
#endif
    generator.generate_bootstrapping_key(
        boot, key, heongpu::ExecutionOptions().set_stream(stream));
    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(boot.is_generated());
#ifdef HEONGPU_TEST_WRAP_RAND_PRIV_BYTES
    EXPECT_GT(random_calls - initial_calls, 1);
#endif
    heongpu::HEEncryptor<S> encryptor(context, key);
    heongpu::HEDecryptor<S> decryptor(context, key);
    heongpu::HELogicOperator<S> logic(context);
    heongpu::Ciphertext<S> first(context), second(context), output(context);
    std::vector<bool> a{false, false, true, true}, b{false, true, false, true};
    encryptor.encrypt(first, a);
    encryptor.encrypt(second, b);
    logic.AND(first, second, output, boot);
    std::vector<bool> actual;
    decryptor.decrypt(output, actual);
    EXPECT_EQ(actual, (std::vector<bool>{false, false, false, true}))
        << "before host transfer";
    boot.store_in_host(stream);
    ASSERT_FALSE(boot.is_on_device());
    boot.store_in_device(stream);
    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    logic.AND(first, second, output, boot);
    decryptor.decrypt(output, actual);
    EXPECT_EQ(actual, (std::vector<bool>{false, false, false, true}))
        << "after host transfer";
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}
