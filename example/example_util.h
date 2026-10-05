// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <string>
#include <vector>
#include <iomanip>
#include <random>

// These examples have been developed with reference to the Microsoft SEAL
// library.

template <typename T>
void display_matrix(const std::vector<T>& matrix, std::size_t row_size)
{
    const std::size_t display_count = 5;

    auto print_row = [&](std::size_t start_index)
    {
        std::cout << "    [";
        for (std::size_t i = 0; i < display_count; ++i)
        {
            std::cout << std::setw(3) << std::right << matrix[start_index + i]
                      << ",";
        }
        std::cout << std::setw(3) << " ...,";
        for (std::size_t i = row_size - display_count; i < row_size; ++i)
        {
            std::cout << std::setw(3) << matrix[start_index + i];
            if (i != row_size - 1)
            {
                std::cout << ",";
            }
            else
            {
                std::cout << " ]\n";
            }
        }
    };

    std::cout << "\n";
    print_row(0);
    print_row(row_size);
    std::cout << std::endl;
}

template <typename T>
inline void display_vector(const std::vector<T>& vec,
                           std::size_t display_count = 4, int precision = 3)
{
    std::ios cout_state(nullptr);
    cout_state.copyfmt(std::cout);

    auto vec_size = vec.size();

    std::cout << std::fixed << std::setprecision(precision);
    std::cout << std::endl;

    if (vec_size <= 2 * display_count)
    {
        std::cout << "    [";
        for (std::size_t i = 0; i < vec_size; ++i)
        {
            std::cout << " " << vec[i] << ((i != vec_size - 1) ? "," : " ]\n");
        }
    }
    else
    {
        std::cout << "    [";
        for (std::size_t i = 0; i < display_count; ++i)
        {
            std::cout << " " << vec[i] << ",";
        }

        std::cout << " ...,";

        for (std::size_t i = vec_size - display_count; i < vec_size; ++i)
        {
            std::cout << " " << vec[i] << ((i != vec_size - 1) ? "," : " ]\n");
        }
    }

    std::cout << std::endl;
    std::cout.copyfmt(cout_state);
}

std::vector<uint64_t> random_vector_generator(int size)
{
    // since it just a example to show default and multi stream usage
    // correctness is not important. that is why we will use random numbers for
    // plaintext
    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = (unsigned long long) 1 << 20;
    unsigned long long maxNumber = ((unsigned long long) 1 << 20) - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);
    unsigned long long number = dis(gen);

    std::uniform_int_distribution<unsigned long long> dis2(0, number);

    std::vector<uint64_t> result;
    for (int i = 0; i < size; i++)
    {
        result.push_back(dis2(gen));
    }

    return result;
}