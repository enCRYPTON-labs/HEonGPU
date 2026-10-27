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
  # An installed GoogleTest is used when find_package finds one.
  FIND_PACKAGE_ARGS NAMES GTest
)

# GoogleTest is only used to build the tests; do not install it with HEonGPU.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

list(APPEND CMAKE_MODULE_PATH ${googletest_SOURCE_DIR}/cmake)
include(GoogleTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/test)

set(EXECUTABLES
    bfv_addition_testcases test_bfv_addition.cpp
    bfv_encoding_testcases test_bfv_encoding.cpp
    bfv_encryption_testcases test_bfv_encryption.cpp
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
    context_randomness_testcases test_context_randomness.cpp
    serialization_context_testcases test_serialization_context.cpp
    serialization_format_testcases test_serialization_format.cpp
    to_string_testcases test_to_string.cpp
    galoiskey_range_testcases test_galoiskey_range.cpp
    rmm_default_testcases test_rmm_default.cpp
    kernel_stack_testcases test_kernel_stack.cpp
    lifecycle_testcases test_lifecycle.cpp
    ckks_encode_depth_testcases test_ckks_encode_depth.cpp
    shortcuts_testcases test_shortcuts.cpp
    linear_helpers_testcases test_linear_helpers.cpp
    polynomial_testcases test_polynomial.cpp
    bootstrapping_presets_testcases test_bootstrapping_presets.cpp
    rotate_hoisted_testcases test_rotate_hoisted.cpp
    ckks_decrypt_depth_testcases test_ckks_decrypt_depth.cpp
    prime_generation_testcases test_prime_generation.cpp
    ckks_parameters_testcases test_ckks_parameters.cpp
    bfv_parameters_testcases test_bfv_parameters.cpp
)

function(add_test exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart GTest::gtest_main)
    if((exe STREQUAL "ckks_secretkey_randomness_testcases" OR exe STREQUAL "secretkey_security_testcases") AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(${exe} PRIVATE HEONGPU_TEST_WRAP_RAND_PRIV_BYTES)
        target_link_libraries(${exe} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    if(exe STREQUAL "ckks_secretkey_randomness_testcases")
        # Uses GPU-NTT directly, which a shared HEonGPU does not export.
        target_link_libraries(${exe} PRIVATE ntt)
    endif()
    if(exe STREQUAL "context_randomness_testcases" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(${exe} PRIVATE HEONGPU_TEST_WRAP_RAND_BYTES)
        target_link_libraries(${exe} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    gtest_discover_tests(${exe}
        PROPERTIES
          LABELS "unit"
          TIMEOUT 30
    )
endfunction()

if(NOT HEonGPU_BUILD_SHARED)
    # Launches a kernel of HEonGPU directly, which is only possible when HEonGPU
    # is linked statically.
    list(APPEND EXECUTABLES bfv_noise_budget_testcases test_bfv_noise_budget.cu)
endif()

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

if(HEonGPU_BUILD_SHARED)
    # Two libraries that use HEonGPU, to check that they share its state.
    foreach(plugin IN ITEMS plugin_a plugin_b)
        add_library(heongpu_test_${plugin} SHARED shared_state_plugin.cpp)
        target_compile_definitions(heongpu_test_${plugin}
            PRIVATE HEONGPU_PLUGIN=${plugin})
        target_link_libraries(heongpu_test_${plugin} PRIVATE heongpu)
    endforeach()
    add_test(shared_state_testcases test_shared_state.cpp)
    target_link_libraries(shared_state_testcases
        PRIVATE heongpu_test_plugin_a heongpu_test_plugin_b)
endif()
