# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(HEONGPU_INCLUDE_PARENT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HEONGPU_INCLUDE_ROOT ${HEONGPU_INCLUDE_PARENT}/heongpu)

# GMP
# We use <gmp.h> in public headers and link against libgmp. Some environments
# don't provide a CMake package that defines GMP::GMP, so create an imported
# target when needed.
if(NOT TARGET GMP::GMP)
    set(_heongpu_gmp_hints "")
    if(DEFINED GMP_ROOT)
        list(APPEND _heongpu_gmp_hints "${GMP_ROOT}")
    endif()
    if(DEFINED ENV{GMP_ROOT})
        list(APPEND _heongpu_gmp_hints "$ENV{GMP_ROOT}")
    endif()

    find_library(GMP_LIBRARY NAMES gmp
        HINTS ${_heongpu_gmp_hints}
        PATH_SUFFIXES lib lib64
    )
    find_path(GMP_INCLUDE_DIR NAMES gmp.h
        HINTS ${_heongpu_gmp_hints}
        PATH_SUFFIXES include
    )

    if(NOT GMP_LIBRARY OR NOT GMP_INCLUDE_DIR)
        message(FATAL_ERROR "GMP not found. Install the GMP development package (e.g. libgmp-dev) or set GMP_ROOT.")
    endif()

    add_library(GMP::GMP UNKNOWN IMPORTED)
    set_target_properties(GMP::GMP PROPERTIES
        IMPORTED_LOCATION "${GMP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
    )
endif()

# NTL
# Used only inside the library (CKKS bootstrapping), so it is linked
# privately. Like GMP, it usually has no CMake package.
if(NOT TARGET NTL::NTL)
    set(_heongpu_ntl_hints "")
    if(DEFINED NTL_ROOT)
        list(APPEND _heongpu_ntl_hints "${NTL_ROOT}")
    endif()
    if(DEFINED ENV{NTL_ROOT})
        list(APPEND _heongpu_ntl_hints "$ENV{NTL_ROOT}")
    endif()

    find_library(NTL_LIBRARY NAMES ntl
        HINTS ${_heongpu_ntl_hints}
        PATH_SUFFIXES lib lib64
    )
    find_path(NTL_INCLUDE_DIR NAMES NTL/ZZ.h
        HINTS ${_heongpu_ntl_hints}
        PATH_SUFFIXES include
    )

    if(NOT NTL_LIBRARY OR NOT NTL_INCLUDE_DIR)
        message(FATAL_ERROR "NTL not found. Install the NTL development package (e.g. libntl-dev) or set NTL_ROOT.")
    endif()

    add_library(NTL::NTL UNKNOWN IMPORTED)
    set_target_properties(NTL::NTL PROPERTIES
        IMPORTED_LOCATION "${NTL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NTL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES GMP::GMP
    )
endif()

set(HEONGPU_UTIL_SOURCES
    lib/util/defaultmodulus.cpp
    lib/util/memorypool.cu
    lib/util/random.cu
    lib/util/serializer.cpp
    lib/util/serialization_header.cpp
    lib/util/util.cu
)

add_library(heongpu_util OBJECT ${HEONGPU_UTIL_SOURCES})
target_compile_features(heongpu_util PUBLIC cxx_std_20)
target_include_directories(heongpu_util
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<BUILD_INTERFACE:${GPUNTT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
)
target_link_libraries(heongpu_util
    PUBLIC
        ntt
        fft
        rngongpu
        rmm::rmm
        GMP::GMP
)
set_target_properties(heongpu_util PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set(HEONGPU_KERNEL_SOURCES
    lib/kernel/addition.cu
    lib/kernel/bootstrapping.cu
    lib/kernel/contextpool.cpp
    lib/kernel/decryption.cu
    lib/kernel/encoding.cu
    lib/kernel/encryption.cu
    lib/kernel/keygeneration.cu
    lib/kernel/multiplication.cu
    lib/kernel/small_ntt.cu
    lib/kernel/switchkey.cu
)

add_library(heongpu_kernel OBJECT ${HEONGPU_KERNEL_SOURCES})
target_compile_features(heongpu_kernel PUBLIC cxx_std_20)
target_include_directories(heongpu_kernel
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
)
target_link_libraries(heongpu_kernel PUBLIC heongpu_util)
set_target_properties(heongpu_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set(HEONGPU_HOST_CORE_SOURCES
    lib/heongpu.cpp
)

add_library(heongpu_host_core OBJECT ${HEONGPU_HOST_CORE_SOURCES})
target_compile_features(heongpu_host_core PUBLIC cxx_std_20)
target_include_directories(heongpu_host_core
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/host>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
)
target_link_libraries(heongpu_host_core PUBLIC heongpu_kernel heongpu_util)
set_target_properties(heongpu_host_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set(HEONGPU_HOST_BFV_SOURCES
    lib/host/bfv/ciphertext.cu
    lib/host/bfv/context.cu
    lib/host/bfv/decryptor.cu
    lib/host/bfv/encoder.cu
    lib/host/bfv/encryptor.cu
    lib/host/bfv/evaluationkey.cu
    lib/host/bfv/keygenerator.cu
    lib/host/bfv/mpcmanager.cu
    lib/host/bfv/operator.cu
    lib/host/bfv/plaintext.cu
    lib/host/bfv/publickey.cu
    lib/host/bfv/secretkey.cu
)

add_library(heongpu_host_bfv OBJECT ${HEONGPU_HOST_BFV_SOURCES})
target_compile_features(heongpu_host_bfv PUBLIC cxx_std_20)
target_include_directories(heongpu_host_bfv
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/host>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
)
target_link_libraries(heongpu_host_bfv PUBLIC heongpu_host_core heongpu_kernel heongpu_util PRIVATE OpenSSL::Crypto)
set_target_properties(heongpu_host_bfv PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set(HEONGPU_HOST_CKKS_SOURCES
    lib/host/ckks/chebyshev_interpolation.cu
    lib/host/ckks/ciphertext.cu
    lib/host/ckks/context.cu
    lib/host/ckks/cosine_approx.cu
    lib/host/ckks/decryptor.cu
    lib/host/ckks/encoder.cu
    lib/host/ckks/encryptor.cu
    lib/host/ckks/evaluationkey.cu
    lib/host/ckks/keygenerator.cu
    lib/host/ckks/mpcmanager.cu
    lib/host/ckks/operator.cu
    lib/host/ckks/plaintext.cu
    lib/host/ckks/precision.cu
    lib/host/ckks/publickey.cu
    lib/host/ckks/secretkey.cu
)

add_library(heongpu_host_ckks OBJECT ${HEONGPU_HOST_CKKS_SOURCES})
target_compile_features(heongpu_host_ckks PUBLIC cxx_std_20)
target_include_directories(heongpu_host_ckks
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/host>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
)
target_link_libraries(heongpu_host_ckks PUBLIC heongpu_host_core heongpu_kernel heongpu_util
    PRIVATE OpenSSL::Crypto NTL::NTL)
set_target_properties(heongpu_host_ckks PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set(HEONGPU_HOST_TFHE_SOURCES
    lib/host/tfhe/ciphertext.cu
    lib/host/tfhe/context.cu
    lib/host/tfhe/decryptor.cu
    lib/host/tfhe/encryptor.cu
    lib/host/tfhe/evaluationkey.cu
    lib/host/tfhe/keygenerator.cu
    lib/host/tfhe/operator.cu
    lib/host/tfhe/secretkey.cu 
)

add_library(heongpu_host_tfhe OBJECT ${HEONGPU_HOST_TFHE_SOURCES})
target_compile_features(heongpu_host_tfhe PUBLIC cxx_std_20)
target_include_directories(heongpu_host_tfhe
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/host>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
)
target_link_libraries(heongpu_host_tfhe PUBLIC heongpu_host_core heongpu_kernel heongpu_util PRIVATE OpenSSL::Crypto)
set_target_properties(heongpu_host_tfhe PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

option(HEonGPU_BUILD_SHARED "Build HEonGPU as a shared library" OFF)
if(HEonGPU_BUILD_SHARED)
    set(_heongpu_library_type SHARED)
else()
    set(_heongpu_library_type STATIC)
endif()

add_library(heongpu ${_heongpu_library_type}
    $<TARGET_OBJECTS:heongpu_util>
    $<TARGET_OBJECTS:heongpu_kernel>
    $<TARGET_OBJECTS:heongpu_host_core>
    $<TARGET_OBJECTS:heongpu_host_bfv>
    $<TARGET_OBJECTS:heongpu_host_ckks>
    $<TARGET_OBJECTS:heongpu_host_tfhe>
)
target_compile_features(heongpu PUBLIC cxx_std_20)

set_target_properties(heongpu PROPERTIES
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME "heongpu"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
if(HEonGPU_BUILD_SHARED)
    # The library does its own device link, so that programs using it need no
    # CUDA compilation. The ABI can change between minor versions.
    set_target_properties(heongpu PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        INSTALL_RPATH "$ORIGIN"
    )
    # Keep the symbols of the static libraries linked into it (GPU-NTT,
    # GPU-FFT, RNGonGPU) private.
    target_link_options(heongpu PRIVATE "LINKER:--exclude-libs,ALL")
endif()

target_include_directories(heongpu
    PUBLIC
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_PARENT}>
        $<INSTALL_INTERFACE:${INCLUDES_INSTALL_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/host>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/kernel>
        $<BUILD_INTERFACE:${HEONGPU_INCLUDE_ROOT}/util>
)

if(HEonGPU_BUILD_SHARED)
    # GPU-NTT, GPU-FFT and RNGonGPU are linked into the shared library;
    # programs only need their headers, which the public headers include.
    target_link_libraries(heongpu
        PUBLIC
            $<COMPILE_ONLY:ntt>
            $<COMPILE_ONLY:fft>
            $<COMPILE_ONLY:rngongpu>
            GMP::GMP
            rmm::rmm
            CUDA::cudart
            CUDA::curand
            OpenSSL::Crypto
            ZLIB::ZLIB
        PRIVATE
            ntt
            fft
            rngongpu
            NTL::NTL
    )
else()
    target_link_libraries(heongpu
        PUBLIC
            ntt
            fft
            rngongpu
            GMP::GMP
            rmm::rmm
            CUDA::curand
            OpenSSL::Crypto
            ZLIB::ZLIB
        PRIVATE
            NTL::NTL
    )
endif()
target_compile_options(heongpu PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    $<$<CONFIG:Debug>:-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
)

install(TARGETS heongpu
  EXPORT ${HEonGPU_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${INCLUDES_INSTALL_DIR}
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cuh"
    PATTERN "*.hpp"
)
