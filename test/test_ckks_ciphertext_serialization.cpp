// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <array>
#include <climits>
#include <limits>
#include <sstream>

namespace
{
    using heongpu::CudaException;
    constexpr auto scheme = heongpu::Scheme::CKKS;
    constexpr int degree = 4096;

    heongpu::HEContext<scheme> make_context(int n = degree, int count = 2)
    {
        auto context =
            heongpu::GenHEContext<scheme>(heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(n);
        std::vector<int> bit_sizes(count, 30);
        bit_sizes[0] = 40;
        context->set_coeff_modulus_bit_sizes(bit_sizes, {40});
        heongpu::MemoryPoolConfig config;
        config.use_memory_pool = false;
        context->generate(config);
        return context;
    }

    // The existing native binary format writes each field without struct
    // padding.
    struct BinaryCiphertext
    {
        heongpu::scheme_type scheme_;
        int ring_size;
        int modulus_count;
        int cipher_size;
        int depth;
        uint8_t ntt;
        heongpu::storage_type storage;
        double scale;
        heongpu::encoding encoding;
        uint8_t rescale;
        uint8_t relinearize;
        uint8_t generated;
        uint32_t size;
        std::string payload;

        explicit BinaryCiphertext(const std::string& binary)
        {
            std::stringstream stream(binary);
            fields(
                [&](auto& value) {
                    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
                });
            if (!stream)
            {
                throw std::runtime_error("Invalid test fixture");
            }
            payload = std::string(std::istreambuf_iterator<char>(stream), {});
        }

        template <typename F> void fields(F function)
        {
            function(scheme_);
            function(ring_size);
            function(modulus_count);
            function(cipher_size);
            function(depth);
            function(ntt);
            function(storage);
            function(scale);
            function(encoding);
            function(rescale);
            function(relinearize);
            function(generated);
            function(size);
        }

        std::string binary()
        {
            std::stringstream stream;
            fields(
                [&](auto& value) {
                    stream.write(reinterpret_cast<const char*>(&value),
                                 sizeof(value));
                });
            stream.write(payload.data(), payload.size());
            return stream.str();
        }
    };

    std::string save(const heongpu::Ciphertext<scheme>& ciphertext)
    {
        std::stringstream stream;
        ciphertext.save(stream);
        return stream.str();
    }

    std::vector<Data64> data(heongpu::Ciphertext<scheme>& ciphertext)
    {
        std::vector<Data64> values;
        ciphertext.get_data(values);
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
        return values;
    }

    class CKKSCiphertextSerialization : public ::testing::Test
    {
      protected:
        void SetUp() override
        {
            context = make_context();
            heongpu::HEKeyGenerator<scheme> generator(context);
            heongpu::Secretkey<scheme> secret(context);
            generator.generate_secret_key(secret);
            heongpu::Publickey<scheme> public_key(context);
            generator.generate_public_key(public_key, secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<scheme>>(
                context, public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<scheme>>(context, secret);
            encoder = std::make_unique<heongpu::HEEncoder<scheme>>(context);
            operators = std::make_unique<heongpu::HEArithmeticOperator<scheme>>(
                context, *encoder);
            relin = std::make_unique<heongpu::Relinkey<scheme>>(context);
            generator.generate_relin_key(*relin, secret);
            ciphertext = std::make_unique<heongpu::Ciphertext<scheme>>(context);
            heongpu::Plaintext<scheme> plain(context);
            message.assign(degree / 2, 0.125);
            encoder->encode(plain, message, std::pow(2.0, 30));
            encryptor->encrypt(*ciphertext, plain);
            binary = save(*ciphertext);
        }

        void check_decryption(heongpu::Ciphertext<scheme>& input,
                              double expected,
                              const heongpu::ExecutionOptions& options = {})
        {
            heongpu::Plaintext<scheme> plain(context);
            decryptor->decrypt(plain, input, options);
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            std::vector<double> decoded;
            encoder->decode(decoded, plain);
            ASSERT_EQ(decoded.size(), message.size());
            for (auto value : decoded)
            {
                EXPECT_NEAR(value, expected, 2e-4);
            }
        }

        void reject_and_retry(const std::string& malformed)
        {
            for (auto storage :
                 {heongpu::storage_type::DEVICE, heongpu::storage_type::HOST})
            {
                heongpu::Ciphertext<scheme> target(
                    context,
                    heongpu::ExecutionOptions().set_storage_type(storage));
                auto* pointer = target.data();
                std::stringstream bad(malformed);
                EXPECT_THROW(target.load(bad), std::exception);
                EXPECT_EQ(target.ring_size(), degree);
                EXPECT_EQ(target.coeff_modulus_count(), 2);
                EXPECT_EQ(target.depth(), 0);
                EXPECT_EQ(target.data(), pointer);
                EXPECT_EQ(target.is_on_device(),
                          storage == heongpu::storage_type::DEVICE);
                EXPECT_THROW(save(target), std::runtime_error);
                std::stringstream good(binary);
                ASSERT_NO_THROW(target.load(good));
                EXPECT_EQ(data(target), data(*ciphertext));
            }
            heongpu::Ciphertext<scheme> unbound;
            std::stringstream bad(malformed);
            EXPECT_THROW(unbound.load(bad), std::exception);
            EXPECT_EQ(unbound.ring_size(), 0);
            EXPECT_THROW(save(unbound), std::runtime_error);
            std::stringstream good(binary);
            EXPECT_NO_THROW(unbound.load(good));
        }

        heongpu::HEContext<scheme> context;
        std::unique_ptr<heongpu::HEEncryptor<scheme>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<scheme>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<scheme>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<scheme>> operators;
        std::unique_ptr<heongpu::Relinkey<scheme>> relin;
        std::unique_ptr<heongpu::Ciphertext<scheme>> ciphertext;
        std::vector<double> message;
        std::string binary;
    };
} // namespace

TEST_F(CKKSCiphertextSerialization, RejectsOriginalOutOfBoundsPayload)
{
    BinaryCiphertext bad(binary);
    bad.ring_size = 1;
    bad.modulus_count = 2;
    bad.cipher_size = 2;
    bad.depth = 0;
    bad.size = 4;
    bad.payload.assign(4 * sizeof(Data64), 0);
    reject_and_retry(bad.binary());
}

TEST_F(CKKSCiphertextSerialization,
       RejectsTruncatedMetadataAndPayloadWithoutMutating)
{
    const size_t header_size =
        binary.size() - BinaryCiphertext(binary).payload.size();
    for (size_t length : {size_t(0), size_t(1), header_size - 1, header_size,
                          binary.size() - 1})
    {
        SCOPED_TRACE(length);
        reject_and_retry(binary.substr(0, length));
    }
    // Streams configured to throw must also leave the destination unchanged.
    heongpu::Ciphertext<scheme> target(context);
    std::stringstream input(binary.substr(0, 1));
    input.exceptions(std::ios::failbit | std::ios::badbit);
    EXPECT_THROW(target.load(input), std::ios_base::failure);
    EXPECT_EQ(target.ring_size(), degree);
    std::stringstream retry(binary);
    EXPECT_NO_THROW(target.load(retry));
}

TEST_F(CKKSCiphertextSerialization, RejectsInvalidDimensionsDepthAndOverflow)
{
    std::vector<BinaryCiphertext> cases;
    auto add = [&](auto edit)
    {
        BinaryCiphertext bad(binary);
        edit(bad);
        cases.push_back(std::move(bad));
    };
    add([](auto& h) { h.ring_size = -1; });
    add([](auto& h) { h.ring_size = degree + 1; });
    add([](auto& h) { h.modulus_count = 0; });
    add([](auto& h) { h.modulus_count = -1; });
    add([](auto& h) { h.depth = -1; });
    add([](auto& h) { h.depth = h.modulus_count; });
    add([](auto& h) { h.depth = INT_MAX; });
    add([](auto& h) { h.cipher_size = 1; });
    add([](auto& h) { h.cipher_size = 4; });
    add([](auto& h) { h.cipher_size = INT_MAX; });
    add(
        [](auto& h)
        {
            h.modulus_count = INT_MAX;
            h.size = UINT32_MAX;
            h.payload.clear();
        });
    add(
        [](auto& h)
        {
            h.size = UINT32_MAX;
            h.payload.clear();
        });
    add([](auto& h) { --h.size; });
    add([](auto& h) { ++h.size; });
    for (auto& bad : cases)
    {
        SCOPED_TRACE(bad.depth);
        reject_and_retry(bad.binary());
    }
}

TEST_F(CKKSCiphertextSerialization, RejectsInvalidEnumsFlagsAndScale)
{
    for (int field = 0; field < 13; ++field)
    {
        SCOPED_TRACE(field);
        BinaryCiphertext bad(binary);
        switch (field)
        {
            case 0:
                bad.scheme_ = heongpu::scheme_type::bfv;
                break;
            case 1:
                bad.storage = static_cast<heongpu::storage_type>(0xff);
                break;
            case 2:
                bad.encoding = static_cast<heongpu::encoding>(0xff);
                break;
            case 3:
                bad.ntt = 2;
                break;
            case 4:
                bad.rescale = 2;
                break;
            case 5:
                bad.relinearize = 2;
                break;
            case 6:
                bad.generated = 2;
                break;
            case 7:
                bad.generated = 0;
                break;
            case 8:
                bad.scale = 0;
                break;
            case 9:
                bad.scale = -1;
                break;
            case 10:
                bad.scale = std::numeric_limits<double>::infinity();
                break;
            case 11:
                bad.scale = std::numeric_limits<double>::quiet_NaN();
                break;
            case 12:
                bad.relinearize = 1;
                break;
        }
        reject_and_retry(bad.binary());
    }
}

TEST_F(CKKSCiphertextSerialization, BoundLoadRejectsContextMismatch)
{
    auto other_degree = make_context(8192);
    auto other_count = make_context(degree, 3);
    for (auto other : {other_degree, other_count})
    {
        heongpu::Ciphertext<scheme> target(other);
        std::stringstream input(binary);
        EXPECT_THROW(target.load(input), std::invalid_argument);
        EXPECT_EQ(target.ring_size(), other->get_poly_modulus_degree());
        EXPECT_EQ(target.coeff_modulus_count(),
                  other->get_ciphertext_modulus_count());
    }
}

TEST_F(CKKSCiphertextSerialization,
       DefaultDeserializeIsCheckedBeforeGPUOperations)
{
    auto larger = make_context(8192);
    heongpu::HEKeyGenerator<scheme> generator(larger);
    heongpu::Secretkey<scheme> secret(larger);
    generator.generate_secret_key(secret);
    heongpu::HEDecryptor<scheme> other_decryptor(larger, secret);
    heongpu::HEEncoder<scheme> other_encoder(larger);
    heongpu::HEArithmeticOperator<scheme> other_operators(larger,
                                                          other_encoder);
    auto loaded = heongpu::serializer::deserialize<heongpu::Ciphertext<scheme>>(
        heongpu::serializer::serialize(*ciphertext));
    heongpu::Plaintext<scheme> plain(larger);
    heongpu::Ciphertext<scheme> result(larger);
    EXPECT_THROW(other_decryptor.decrypt(plain, loaded), std::invalid_argument);
    EXPECT_THROW(other_operators.add(loaded, loaded, result),
                 std::invalid_argument);
    EXPECT_THROW(other_operators.multiply(loaded, loaded, result),
                 std::invalid_argument);
    EXPECT_THROW(other_operators.negate(loaded, result), std::invalid_argument);
    EXPECT_THROW(other_operators.mod_drop(loaded, result),
                 std::invalid_argument);
    heongpu::Ciphertext<scheme> copied(loaded), moved(std::move(copied));
    EXPECT_THROW(other_decryptor.decrypt(plain, moved), std::invalid_argument);
    EXPECT_THROW(decryptor->decrypt(plain, copied), std::logic_error);
    check_decryption(loaded, 0.125);
}

TEST_F(CKKSCiphertextSerialization, SlotCoefficientAndHostRoundTripsWork)
{
    for (bool coefficient : {false, true})
    {
        heongpu::Plaintext<scheme> plain(context);
        heongpu::Ciphertext<scheme> source(context);
        if (coefficient)
        {
            std::vector<double> values(degree, 0);
            values[0] = 7;
            values[23] = -11;
            encoder->encode(plain, values, std::pow(2.0, 30),
                            heongpu::ExecutionOptions(),
                            heongpu::encoding::COEFFICIENT);
        }
        else
        {
            encoder->encode(plain, message, std::pow(2.0, 30));
        }
        encryptor->encrypt(source, plain);
        source.store_in_host();
        const auto compressed = heongpu::serializer::serialize(source);
        auto restored =
            heongpu::serializer::deserialize<heongpu::Ciphertext<scheme>>(
                compressed);
        EXPECT_EQ(data(source), data(restored));
        EXPECT_EQ(source.encoding_type(), restored.encoding_type());
        heongpu::Plaintext<scheme> decoded(context);
        decryptor->decrypt(decoded, restored);
        if (coefficient)
        {
            std::vector<double> values;
            encoder->decode(values, decoded);
            EXPECT_NEAR(values[0], 7, 2e-4);
            EXPECT_NEAR(values[23], -11, 2e-4);
        }
        else
        {
            check_decryption(restored, 0.125);
        }
    }
}

TEST_F(CKKSCiphertextSerialization,
       RescaledAndUnrelinearizedCiphertextsRoundTrip)
{
    heongpu::Ciphertext<scheme> product(context);
    operators->multiply(*ciphertext, *ciphertext, product);
    ASSERT_EQ(product.size(), 3);
    heongpu::Ciphertext<scheme> restored(context);
    std::stringstream multiplied(save(product));
    ASSERT_NO_THROW(restored.load(multiplied));
    EXPECT_EQ(data(product), data(restored));
    heongpu::Plaintext<scheme> plain(context);
    EXPECT_THROW(decryptor->decrypt(plain, restored), std::invalid_argument);
    operators->relinearize_inplace(restored, *relin);
    ASSERT_EQ(restored.size(), 2);
    heongpu::Ciphertext<scheme> relinearized(context);
    std::stringstream after_relinearization(save(restored));
    ASSERT_NO_THROW(relinearized.load(after_relinearization));
    EXPECT_EQ(data(relinearized).size(), size_t(2 * degree * 2));
    operators->rescale_inplace(relinearized);
    restored = std::move(relinearized);
    ASSERT_EQ(restored.depth(), 1);
    heongpu::Ciphertext<scheme> leveled(context);
    std::stringstream rescaled(save(restored));
    ASSERT_NO_THROW(leveled.load(rescaled));
    check_decryption(leveled, 0.015625);
}

TEST_F(CKKSCiphertextSerialization,
       LoadPreservesDestinationStreamAndCompletedData)
{
    cudaStream_t stream;
    HEONGPU_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    heongpu::Ciphertext<scheme> restored(
        context, heongpu::ExecutionOptions().set_stream(stream));
    std::stringstream input(binary);
    ASSERT_NO_THROW(restored.load(input));
    EXPECT_EQ(restored.stream(), stream);
    check_decryption(restored, 0.125,
                     heongpu::ExecutionOptions().set_stream(stream));
    restored.store_in_host(stream);
    EXPECT_EQ(data(restored), data(*ciphertext));
    HEONGPU_CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_F(CKKSCiphertextSerialization, ExistingCiphertextCannotBeOverwrittenByLoad)
{
    const auto before = data(*ciphertext);
    std::stringstream input(binary);
    EXPECT_THROW(ciphertext->load(input), std::runtime_error);
    EXPECT_EQ(data(*ciphertext), before);
    check_decryption(*ciphertext, 0.125);
}

TEST_F(CKKSCiphertextSerialization, LastModulusCannotBeRescaled)
{
    operators->mod_drop_inplace(*ciphertext);
    BinaryCiphertext header(save(*ciphertext));
    header.rescale = 1;
    heongpu::Ciphertext<scheme> restored(context);
    std::stringstream input(header.binary());
    ASSERT_NO_THROW(restored.load(input));
    EXPECT_THROW(operators->rescale_inplace(restored), std::logic_error);
    EXPECT_EQ(restored.depth(), 1);
}

TEST_F(CKKSCiphertextSerialization, MultipartyRejectsMismatchedAndShortInputs)
{
    auto larger = make_context(8192);
    heongpu::HEEncoder<scheme> other_encoder(larger);
    double scale = std::pow(2.0, 30);
    heongpu::HEMultiPartyManager<scheme> other_manager(larger, other_encoder,
                                                       scale);
    heongpu::HEKeyGenerator<scheme> generator(larger);
    heongpu::Secretkey<scheme> secret(larger);
    generator.generate_secret_key(secret);
    heongpu::Ciphertext<scheme> output(larger);
    heongpu::Plaintext<scheme> plain(larger);
    std::vector<heongpu::Ciphertext<scheme>> shares{*ciphertext};
    EXPECT_THROW(other_manager.decrypt_partial(*ciphertext, secret, output),
                 std::invalid_argument);
    EXPECT_THROW(other_manager.decrypt(shares, plain), std::invalid_argument);
    heongpu::RNGSeed seed;
    EXPECT_THROW(other_manager.distributed_bootstrapping_participant(
                     *ciphertext, output, secret, seed),
                 std::invalid_argument);
    EXPECT_THROW(other_manager.distributed_bootstrapping_coordinator(
                     shares, *ciphertext, output, seed),
                 std::invalid_argument);

    heongpu::HEMultiPartyManager<scheme> manager(context, *encoder, scale);
    heongpu::Ciphertext<scheme> deep(*ciphertext);
    operators->mod_drop_inplace(deep);
    heongpu::Ciphertext<scheme> loaded(context), result(context);
    std::stringstream input(save(deep));
    ASSERT_NO_THROW(loaded.load(input));
    shares = {loaded};
    EXPECT_THROW(manager.distributed_bootstrapping_coordinator(shares, deep,
                                                               result, seed),
                 std::invalid_argument);
    shares.clear();
    EXPECT_THROW(manager.distributed_bootstrapping_coordinator(shares, deep,
                                                               result, seed),
                 std::invalid_argument);
}
