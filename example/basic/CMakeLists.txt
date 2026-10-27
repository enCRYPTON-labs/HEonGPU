# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples/basic)

set(EXECUTABLES
    1_basic_bfv 1_basic_bfv.cpp
    2_basic_ckks 2_basic_ckks.cpp
    3_basic_memorypool_config 3_basic_memorypool_config.cpp
    4_switchkey_methods_bfv 4_switchkey_methods_bfv.cpp
    5_switchkey_methods_ckks 5_switchkey_methods_ckks.cpp
    6_ckks_coefficient_encoding 6_ckks_coefficient_encoding.cpp
    7_ckks_coeff_to_slot_roundtrip 7_ckks_coeff_to_slot_roundtrip.cpp
    8_default_stream_usage 8_default_stream_usage.cpp
    9_multi_stream_usage_way1 9_multi_stream_usage_way1.cpp
    10_multi_stream_usage_way2 10_multi_stream_usage_way2.cpp
    11_basic_bfv_logic 11_basic_bfv_logic.cpp
    12_basic_ckks_logic 12_basic_ckks_logic.cpp
    13_bfv_serialization 13_bfv_serialization.cpp
    14_ckks_serialization 14_ckks_serialization.cpp
    15_basic_tfhe 15_basic_tfhe.cpp
    16_ckks_automatic_rescale 16_ckks_automatic_rescale.cpp
    17_ckks_session 17_ckks_session.cpp
    18_bfv_session 18_bfv_session.cpp
)

function(add_example exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
    target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
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
    add_example(${exe} ${source})
endforeach()
