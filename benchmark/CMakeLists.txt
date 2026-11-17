# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/benchmark)

set(EXECUTABLES
    bfv_benchmark benchmark_bfv.cpp
    ckks_benchmark benchmark_ckks.cpp
    tfhe_benchmark benchmark_tfhe.cpp
)

function(add_benchmark exe source)
    add_executable(${exe} ${source})
    if(USE_HIP)
        # Benchmark sources include HEonGPU headers that use __umul64hi and
        # warpSize in non-template device functions (util.cuh,
        # bigintegerarith.cuh). The HIP headers declare those only when
        # a source is compiled as HIP.
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
        target_link_libraries(${exe} PRIVATE heongpu hip::host)
        target_include_directories(${exe} SYSTEM BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/hip_compat)
        target_compile_definitions(${exe} PRIVATE USE_HIP)
        set_target_properties(${exe} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        )
    else()
        target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart)
        set_target_properties(${exe} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    endif()
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_benchmark(${exe} ${source})
endforeach()
