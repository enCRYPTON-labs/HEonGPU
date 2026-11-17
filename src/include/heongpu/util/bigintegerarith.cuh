// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BIGINTEGER_ARITHMATIC_H
#define HEONGPU_BIGINTEGER_ARITHMATIC_H

#include "gpuntt/common/modular_arith.cuh"

namespace heongpu
{
    namespace biginteger
    {

        /*
         * result = 1  ==> number1 > number2
         * result = 0  ==> number1 = number2
         * result = -1 ==> number1 < number2
         */

        __device__ __forceinline__ int min_(int a, int b)
        {
            return (a < b) ? a : b;
        } // TODO: Fix it

        __device__ __forceinline__ int compare(Data64* number1, Data64* number2,
                                               int wordsize)
        {
            int result = 0;
            number1 += wordsize - 1;
            number2 += wordsize - 1;

            while ((result == 0) && wordsize--)
            {
                result = (*number1 > *number2) - (*number1 < *number2);
                --number1;
                --number2;
            }

            return result;
        }

        __device__ __forceinline__ bool
        is_greater(Data64* number1, Data64* number2, int wordsize)
        {
            return compare(number1, number2, wordsize) > 0;
        }

        __device__ __forceinline__ bool
        is_greater_or_equal(Data64* number1, Data64* number2, int wordsize)
        {
            return compare(number1, number2, wordsize) >= 0;
        }

        __device__ __forceinline__ bool is_less(Data64* number1,
                                                Data64* number2, int wordsize)
        {
            return compare(number1, number2, wordsize) < 0;
        }

        __device__ __forceinline__ bool
        is_less_or_equal(Data64* number1, Data64* number2, int wordsize)
        {
            return compare(number1, number2, wordsize) <= 0;
        }

        __device__ __forceinline__ bool is_equal(Data64* number1,
                                                 Data64* number2, int wordsize)
        {
            return compare(number1, number2, wordsize) == 0;
        }

        __device__ __forceinline__ int add(Data64* number1, Data64* number2,
                                           int wordsize, Data64* result)
        {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
            // HIP path: use plain C++ with overflow detection
            Data64 carry = 0;
            for (int i = 0; i < wordsize; i++) {
                Data64 a = number1[i];
                Data64 b = number2[i];
                Data64 sum = a + b + carry;
                // Detect overflow: sum < a when (a + carry) overflows, or sum < b when b overflows
                carry = (sum < a || (carry && sum == a)) ? 1 : 0;
                result[i] = sum;
            }
            return static_cast<int>(carry);
#else
            // CUDA path: PTX inline assembly
            int carry = 0;

            asm("add.cc.u64 %0, %1, %2;"
                : "=l"(result[0])
                : "l"(number1[0]), "l"(number2[0]));
            for (int i = 1; i < wordsize; i++)
                asm("addc.cc.u64 %0, %1, %2;"
                    : "=l"(result[i])
                    : "l"(number1[i]), "l"(number2[i]));

            // asm("addc.u32 %0, %0, %1;" : "+r"(carry) : "r"(0));
            asm("addc.u32 %0, 0, 0;" : "=r"(carry));

            return carry;
#endif
        }

        __device__ __forceinline__ int
        add_inplace(Data64* number1, Data64* number2, int wordsize)
        {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
            // HIP path: use plain C++ with overflow detection
            Data64 carry = 0;
            for (int i = 0; i < wordsize; i++) {
                Data64 a = number1[i];
                Data64 b = number2[i];
                Data64 sum = a + b + carry;
                carry = (sum < a || (carry && sum == a)) ? 1 : 0;
                number1[i] = sum;
            }
            return static_cast<int>(carry);
#else
            // CUDA path: PTX inline assembly
            int carry = 0;

            asm("add.cc.u64 %0, %0, %1;" : "+l"(number1[0]) : "l"(number2[0]));
            for (int i = 1; i < wordsize; i++)
                asm("addc.cc.u64 %0, %0, %1;"
                    : "+l"(number1[i])
                    : "l"(number2[i]));

            // asm("addc.u32 %0, %0, %1;" : "+r"(carry) : "r"(0));
            asm("addc.u32 %0, 0, 0;" : "=r"(carry));

            return carry;
#endif
        }

        __device__ __forceinline__ int sub(Data64* number1, Data64* number2,
                                           int wordsize, Data64* result)
        {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
            // HIP path: use plain C++ with borrow detection
            Data64 borrow = 0;
            for (int i = 0; i < wordsize; i++) {
                Data64 a = number1[i];
                Data64 b = number2[i];
                Data64 diff = a - b - borrow;
                // Borrow occurs if a < b, or a == b and there was a borrow
                borrow = (a < b || (a == b && borrow)) ? 1 : 0;
                result[i] = diff;
            }
            return static_cast<int>(borrow);
#else
            // CUDA path: PTX inline assembly
            int borrow = 0;

            asm("sub.cc.u64 %0, %1, %2;"
                : "=l"(result[0])
                : "l"(number1[0]), "l"(number2[0]));
            for (int i = 1; i < wordsize; i++)
                asm("subc.cc.u64 %0, %1, %2;"
                    : "=l"(result[i])
                    : "l"(number1[i]), "l"(number2[i]));

            asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

            return borrow;
#endif
        }

        __device__ __forceinline__ void sub2(Data64* number1, Data64* number2,
                                             int wordsize, Data64* result)
        {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
            // HIP path: use plain C++ with borrow detection
            Data64 borrow = 0;
            for (int i = 0; i < wordsize; i++) {
                Data64 a = number1[i];
                Data64 b = number2[i];
                Data64 diff = a - b - borrow;
                borrow = (a < b || (a == b && borrow)) ? 1 : 0;
                result[i] = diff;
            }
#else
            // CUDA path: PTX inline assembly
            asm("sub.cc.u64 %0, %1, %2;"
                : "=l"(result[0])
                : "l"(number1[0]), "l"(number2[0]));
            for (int i = 1; i < wordsize; i++)
                asm("subc.cc.u64 %0, %1, %2;"
                    : "=l"(result[i])
                    : "l"(number1[i]), "l"(number2[i]));
#endif
        }

        /*
         * number1 ==> big integer with number1_wordsize 64 bit word size
         * number2 ==> 64 bit number
         * result  ==> big integer with result_wordsize 64 bit word size
         */
        __device__ __forceinline__ void multiply(Data64* number1,
                                                 int number1_wordsize,
                                                 Data64 number2, Data64* result,
                                                 int result_wordsize)
        {
            // set result as zero
            for (int i = 0; i < result_wordsize; i++)
            {
                result[i] = 0;
            }

            int wordsize =
                min_(number1_wordsize, result_wordsize); // TODO: Fix it

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
            // HIP path: use __umul64hi intrinsic for high part
            // Compute low parts first
            for (int i = 0; i < wordsize; i++)
            {
                result[i] = number1[i] * number2;  // low 64 bits
            }

            // Add high parts with carry chain
            Data64 carry = 0;
            for (int i = 0; i < wordsize; i++)
            {
                Data64 hi = __umul64hi(number1[i], number2);
                if (i + 1 < result_wordsize)
                {
                    Data64 prev = result[i + 1];
                    Data64 sum = prev + hi + carry;
                    carry = (sum < prev || (carry && sum == prev)) ? 1 : 0;
                    result[i + 1] = sum;
                }
            }
#else
            // CUDA path: PTX inline assembly
            for (int i = 0; i < wordsize; i++)
            {
                asm("mul.lo.u64 %0, %1, %2;"
                    : "=l"(result[i])
                    : "l"(number1[i]), "l"(number2));
            }

            asm("mad.hi.cc.u64  %0, %1, %2, %0;"
                : "+l"(result[1])
                : "l"(number1[0]), "l"(number2));
            for (int i = 1; i < wordsize - 1; i++)
            {
                asm("madc.hi.cc.u64  %0, %1, %2, %0;"
                    : "+l"(result[i + 1])
                    : "l"(number1[i]), "l"(number2));
            }

            if (number1_wordsize < result_wordsize)
                asm("madc.hi.u64  %0, %1, %2, %0;"
                    : "+l"(result[wordsize])
                    : "l"(number1[wordsize - 1]), "l"(number2));
#endif
        }

        __device__ __forceinline__ void set_zero(Data64* number,
                                                 int number_wordsize)
        {
            // set result as zero
            for (int i = 0; i < number_wordsize; i++)
            {
                number[i] = 0;
            }
        }

        __device__ __forceinline__ void
        set(Data64* number_in, int number_wordsize, Data64* number_out)
        {
            // set result as zero
            for (int i = 0; i < number_wordsize; i++)
            {
                number_out[i] = number_in[i];
            }
        }

    } // namespace biginteger

} // namespace heongpu

#endif // HEONGPU_BIGINTEGER_ARITHMATIC_H
