// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>

#include <string>
#include <iomanip>
#include <omp.h>

constexpr auto Scheme = heongpu::Scheme::BFV;

int main(int argc, char* argv[])
{
    std::vector<size_t> poly_modulus_degrees = {4096, 8192, 16384, 32768,
                                                65536};
    std::vector<int> plain_modulus = {1032193, 1032193, 786433, 786433, 786433};

    int repeat_count = 10;

    for (int i = 0; i < poly_modulus_degrees.size(); i++)
    {
        heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();
        context->set_poly_modulus_degree(poly_modulus_degrees[i]);
        context->set_coeff_modulus_default_values(1);
        context->set_plain_modulus(plain_modulus[i]);
        context->generate();

        heongpu::HEKeyGenerator<Scheme> keygen(context);
        heongpu::Secretkey<Scheme> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<Scheme> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey<Scheme> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        std::vector<int> custom_key_index = {1};
        heongpu::Galoiskey<Scheme> galois_key(context, custom_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        heongpu::HEEncoder<Scheme> encoder(context);
        heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
        heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
        heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

        heongpu::HostVector<uint64_t> message(poly_modulus_degrees[i], 2);

        float time = 0;
        float time_encode = 0;
        float time_encryption = 0;
        float time_addition = 0;
        float time_subtraction = 0;
        float time_multiplication = 0;
        float time_relinearization = 0;
        float time_plainaddition = 0;
        float time_plainsubtraction = 0;
        float time_plainmultiplication = 0;
        float time_rotaterow = 0;
        float time_rotatecolumn = 0;
        float time_decryption = 0;
        float time_decode = 0;

        cudaEvent_t start_time, stop_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&stop_time);

        for (int j = 0; j < repeat_count; j++)
        {
            heongpu::Plaintext<Scheme> P1(context);

            cudaEventRecord(start_time);
            encoder.encode(P1, message);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_encode += time;

            heongpu::Ciphertext<Scheme> C1(context);

            cudaEventRecord(start_time);
            encryptor.encrypt(C1, P1);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_encryption += time;

            heongpu::Ciphertext<Scheme> C2(context);

            cudaEventRecord(start_time);
            operators.add(C1, C1, C2);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_addition += time;

            cudaEventRecord(start_time);
            operators.sub(C2, C1, C2);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_subtraction += time;

            cudaEventRecord(start_time);
            operators.multiply(C2, C1, C2);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_multiplication += time;

            cudaEventRecord(start_time);
            operators.relinearize_inplace(C2, relin_key);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_relinearization += time;

            cudaEventRecord(start_time);
            operators.rotate_rows(C2, C2, galois_key, 1);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_rotaterow += time;

            cudaEventRecord(start_time);
            operators.rotate_columns(C2, C2, galois_key);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_rotatecolumn += time;

            heongpu::Ciphertext<Scheme> C3(context);
            encryptor.encrypt(C3, P1);

            cudaEventRecord(start_time);
            operators.add_plain_inplace(C3, P1);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_plainaddition += time;

            cudaEventRecord(start_time);
            operators.sub_plain_inplace(C3, P1);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_plainsubtraction += time;

            heongpu::Ciphertext<Scheme> C4(context);
            encryptor.encrypt(C4, P1);

            cudaEventRecord(start_time);
            operators.multiply_plain(C4, P1, C4);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_plainmultiplication += time;

            heongpu::Plaintext<Scheme> P2(context);

            cudaEventRecord(start_time);
            decryptor.decrypt(P2, C4);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_decryption += time;

            heongpu::HostVector<uint64_t> mesagge2;
            cudaEventRecord(start_time);
            encoder.decode(mesagge2, P2);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_decode += time;
        }

        std::cout << "============== Benchmark BFV with poly_modulus_degrees: "
                  << poly_modulus_degrees[i]
                  << " and default parameters ==============" << std::endl;
        std::cout << "Average encode timing: " << (time_encode / repeat_count)
                  << " ms" << std::endl;
        std::cout << "Average encryption timing: "
                  << (time_encryption / repeat_count) << " ms" << std::endl;
        std::cout << "Average addition timing: "
                  << (time_addition / repeat_count) << " ms" << std::endl;
        std::cout << "Average subtraction timing: "
                  << (time_subtraction / repeat_count) << " ms" << std::endl;
        std::cout << "Average multiplication timing: "
                  << (time_multiplication / repeat_count) << " ms" << std::endl;
        std::cout << "Average relinearization timing: "
                  << (time_relinearization / repeat_count) << " ms"
                  << std::endl;
        std::cout << "Average plain addition timing: "
                  << (time_plainaddition / repeat_count) << " ms" << std::endl;
        std::cout << "Average plain subtraction timing: "
                  << (time_plainsubtraction / repeat_count) << " ms"
                  << std::endl;
        std::cout << "Average plain multiplication timing: "
                  << (time_plainmultiplication / repeat_count) << " ms"
                  << std::endl;
        std::cout << "Average rotate row timing: "
                  << (time_rotaterow / repeat_count) << " ms" << std::endl;
        std::cout << "Average rotate column timing: "
                  << (time_rotatecolumn / repeat_count) << " ms" << std::endl;
        std::cout << "Average decryption timing: "
                  << (time_decryption / repeat_count) << " ms" << std::endl;
        std::cout << "Average decode timing: " << (time_decode / repeat_count)
                  << " ms" << std::endl;
        std::cout << std::endl << std::endl;
    }

    return EXIT_SUCCESS;
}
