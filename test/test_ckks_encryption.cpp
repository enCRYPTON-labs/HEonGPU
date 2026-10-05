// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

template <typename T>
bool fix_point_equal(T input1, T input2, T epsilon = static_cast<T>(1e-4))
{
    return std::fabs(input1 - input2) < epsilon;
}

template <typename T>
bool fix_point_array_check(const std::vector<T>& array1,
                           const std::vector<T>& array2,
                           T epsilon = static_cast<T>(1e-4))
{
    if (array1.size() != array2.size())
    {
        return false;
    }

    for (size_t i = 0; i < array1.size(); ++i)
    {
        if (!fix_point_equal(array1[i], array2[i], epsilon))
        {
            return false;
        }
    }

    return true;
}

TEST(HEonGPU, CKKS_Encryption_Decryption)
{
    {
        size_t poly_modulus_degree = 4096;
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({40, 30, 30}, {40});
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
        double scale = pow(2.0, 30);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 8192;
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({40, 30, 30, 30, 30}, {40});
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
        double scale = pow(2.0, 30);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 16384;
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35}, {45});
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
        double scale = pow(2.0, 35);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 32768;
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({59, 40, 40, 40, 40, 40, 40, 40,
                                              40, 40, 40, 40, 40, 40, 40, 40,
                                              40, 40, 40},
                                             {59});
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
        double scale = pow(2.0, 40);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 65536;
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {59, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
            {59});
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
        double scale = pow(2.0, 45);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);
    }

    cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}