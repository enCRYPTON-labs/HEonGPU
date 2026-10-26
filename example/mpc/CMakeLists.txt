# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples/mpc)

set(EXECUTABLES
    1_multiparty_computation_bfv 1_multiparty_computation_bfv.cpp
    2_multiparty_computation_ckks 2_multiparty_computation_ckks.cpp
    3_mpc_collective_bootstrapping_bfv 3_mpc_collective_bootstrapping_bfv.cpp
    4_mpc_collective_bootstrapping_ckks 4_mpc_collective_bootstrapping_ckks.cpp
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
