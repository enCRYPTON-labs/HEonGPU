# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.0
  SOURCE_DIR     "${CMAKE_BINARY_DIR}/googletest-src"
)

FetchContent_MakeAvailable(googletest)

list(APPEND CMAKE_MODULE_PATH ${googletest_SOURCE_DIR}/cmake)
include(GoogleTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/test)

set(EXECUTABLES
    bfv_addition_testcases test_bfv_addition.cpp
    bfv_encoding_testcases test_bfv_encoding.cpp
    bfv_encryption_testcases test_bfv_encryption.cpp
    bfv_noise_budget_testcases test_bfv_noise_budget.cu
    bfv_multiplication_testcases test_bfv_multiplication.cpp
    bfv_relinearization_testcases test_bfv_relinearization.cpp
    bfv_rotation_method_1_testcases test_bfv_rotation_method_1.cpp
    bfv_rotation_method_2_testcases test_bfv_rotation_method_2.cpp

    ckks_addition_testcases test_ckks_addition.cpp
    ckks_encoding_testcases test_ckks_encoding.cpp
    ckks_encryption_testcases test_ckks_encryption.cpp
    ckks_secretkey_randomness_testcases test_ckks_secretkey_randomness.cpp
    ckks_ciphertext_serialization_testcases test_ckks_ciphertext_serialization.cpp
    ckks_multiplication_testcases test_ckks_multiplication.cpp
    ckks_relinearization_testcases test_ckks_relinearization.cpp
    ckks_rotation_method_1_testcases test_ckks_rotation_method_1.cpp
    ckks_rotation_method_2_testcases test_ckks_rotation_method_2.cpp
    ckks_galoiskey_testcases test_ckks_galoiskey.cpp

    secretkey_security_testcases test_secretkey_security.cpp
    tfhe_gate_boot_testcases test_tfhe_gate_boot.cpp

    memory_pool_testcases test_memory_pool.cpp

    errors_testcases test_errors.cpp
    context_identity_testcases test_context_identity.cpp
)

function(add_test exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart gtest_main)
    if((exe STREQUAL "ckks_secretkey_randomness_testcases" OR exe STREQUAL "secretkey_security_testcases") AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(${exe} PRIVATE HEONGPU_TEST_WRAP_RAND_PRIV_BYTES)
        target_link_options(${exe} PRIVATE "-Wl,--wrap=RAND_priv_bytes")
    endif()
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    gtest_discover_tests(${exe}
        PROPERTIES
          LABELS "unit"
          TIMEOUT 30
    )
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_test(${exe} ${source})
endforeach()
