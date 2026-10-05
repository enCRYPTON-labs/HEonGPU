// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

TEST(HEonGPU, BFV_Ciphertext_Rotation_Keyswitching_Method_I)
{
    {
        size_t poly_modulus_degree = 4096;
        int plain_modulus = 1032193;
        heongpu::HEContext<heongpu::Scheme::BFV> context =
            heongpu::GenHEContext<heongpu::Scheme::BFV>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({40, 40}, {40});
        context->set_plain_modulus(plain_modulus);
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context,
                                                             public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context,
                                                             secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context,
                                                                      encoder);

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context,
                                                            shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext<heongpu::Scheme::BFV> P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 8192;
        int plain_modulus = 1032193;
        heongpu::HEContext<heongpu::Scheme::BFV> context =
            heongpu::GenHEContext<heongpu::Scheme::BFV>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({54, 54, 54}, {55});
        context->set_plain_modulus(plain_modulus);
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context,
                                                             public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context,
                                                             secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context,
                                                                      encoder);

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context,
                                                            shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext<heongpu::Scheme::BFV> P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 16384;
        int plain_modulus = 786433;
        heongpu::HEContext<heongpu::Scheme::BFV> context =
            heongpu::GenHEContext<heongpu::Scheme::BFV>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes({54, 54, 54, 54, 55, 55, 55},
                                             {55});
        context->set_plain_modulus(plain_modulus);
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context,
                                                             public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context,
                                                             secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context,
                                                                      encoder);

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context,
                                                            shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext<heongpu::Scheme::BFV> P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 32768;
        int plain_modulus = 786433;
        heongpu::HEContext<heongpu::Scheme::BFV> context =
            heongpu::GenHEContext<heongpu::Scheme::BFV>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59}, {59});
        context->set_plain_modulus(plain_modulus);
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context,
                                                             public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context,
                                                             secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context,
                                                                      encoder);

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context,
                                                            shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext<heongpu::Scheme::BFV> P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    cudaDeviceSynchronize();

    {
        size_t poly_modulus_degree = 65536;
        int plain_modulus = 786433;
        // TODO: find optimal way to store huge galois key, maybe store it in
        // CPU RAM.
        // heongpu::HEContext<heongpu::Scheme::BFV> context =
        // heongpu::GenHEContext<heongpu::Scheme::BFV>(
        // heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        // heongpu::sec_level_type::none);
        // context->set_poly_modulus_degree(poly_modulus_degree);
        // context->set_coeff_modulus_bit_sizes({58, 58, 58, 58, 58, 58, 58, 58,
        // 58, 59, 59, 59, 59, 59, 59,
        //     59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59},
        //    {59});
        // context->set_plain_modulus(plain_modulus);
        // context->generate();
        heongpu::HEContext<heongpu::Scheme::BFV> context =
            heongpu::GenHEContext<heongpu::Scheme::BFV>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59}, {59});
        context->set_plain_modulus(plain_modulus);
        context->generate();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context,
                                                             public_key);
        heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context,
                                                             secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context,
                                                                      encoder);

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context,
                                                            shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext<heongpu::Scheme::BFV> P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}