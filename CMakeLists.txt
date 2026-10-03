# Copyright 2024-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(HEonGPU VERSION 1.1.4 LANGUAGES C CXX CUDA ASM)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build HEonGPU version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ Standard is: ${CMAKE_CXX_STANDARD}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto SSL)

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
if(OpenMP_CXX_VERSION)
    message(STATUS "OpenMP Version: ${OpenMP_CXX_VERSION}")
endif()
if(ZLIB_VERSION_STRING)
    message(STATUS "ZLIB Version: ${ZLIB_VERSION_STRING}")
endif()
if(OPENSSL_VERSION)
    message(STATUS "OpenSSL Version: ${OPENSSL_VERSION}")
endif()

set(HEONGPU_CUDA_ARCH_AUTO_DETECTED OFF CACHE BOOL "Set to ON when HEonGPU auto-detects CUDA arch")
set(HEONGPU_CUDA_ARCH_FORCE_MANUAL OFF CACHE BOOL "Set to ON to prevent HEonGPU from auto-detecting CUDA arch")
if(NOT HEONGPU_CUDA_ARCH_FORCE_MANUAL AND (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52" OR HEONGPU_CUDA_ARCH_AUTO_DETECTED))
    include(cmake/DetectCudaArch.cmake)
    heongpu_detect_cuda_arch(HEONGPU_DETECTED_CUDA_ARCH)
    if(HEONGPU_DETECTED_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "${HEONGPU_DETECTED_CUDA_ARCH}" CACHE STRING "CUDA GPU architectures" FORCE)
        set(HEONGPU_CUDA_ARCH_AUTO_DETECTED ON CACHE BOOL "Set to ON when HEonGPU auto-detects CUDA arch" FORCE)
        message(STATUS "Auto-detected CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} for HEonGPU")
    else()
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures" FORCE)
        set(HEONGPU_CUDA_ARCH_AUTO_DETECTED ON CACHE BOOL "Set to ON when HEonGPU auto-detects CUDA arch" FORCE)
        message(WARNING "CUDA arch auto-detection failed; falling back to CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    set(HEONGPU_CUDA_ARCH_AUTO_DETECTED OFF CACHE BOOL "Set to ON when HEonGPU auto-detects CUDA arch" FORCE)
    message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for HEonGPU")
endif()

##################
# Export Library #
##################

set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(HEonGPU_INSTALL_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/HEonGPU-${HEonGPU_INSTALL_VERSION})

set(HEonGPU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(HEonGPU_VERSION_CONFIG "${HEonGPU_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(HEonGPU_PROJECT_CONFIG "${HEonGPU_GENERATED_DIR}/${PROJECT_NAME}Config.cmake")
set(HEonGPU_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(HEonGPU_CONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}-${HEonGPU_INSTALL_VERSION}")
set(HEonGPU_NAMESPACE "${PROJECT_NAME}::")
set(HEonGPU_VERSION ${PROJECT_VERSION})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${HEonGPU_VERSION_CONFIG}" VERSION ${HEonGPU_VERSION} COMPATIBILITY SameMajorVersion
)
configure_file("${HEonGPU_SOURCE_DIR}/cmake/Config.cmake.in" "${HEonGPU_PROJECT_CONFIG}" @ONLY)

install(
    FILES "${HEonGPU_PROJECT_CONFIG}" "${HEonGPU_VERSION_CONFIG}"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

install(
    FILES "${HEonGPU_SOURCE_DIR}/cmake/FindCCCL.cmake"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

install(
    EXPORT "${HEonGPU_TARGETS_EXPORT_NAME}"
    NAMESPACE "${HEonGPU_NAMESPACE}"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

# Removed duplicate -O3 flag - optimization is handled via target_compile_options in src/CMakeLists.txt

set(RNGonGPU_USE_INTERNAL_GPUNTT
    OFF CACHE BOOL
    "Force using global GPU-NTT (thirdparty/GPU-NTT) instead of RNGonGPU's internal" FORCE)

add_subdirectory(thirdparty)
add_subdirectory(src)

############################################################################
# Build Options
############################################################################

option(HEonGPU_BUILD_TESTS "Build HEonGPU Tests" OFF)
option(HEonGPU_BUILD_EXAMPLES "Build HEonGPU Examples" OFF)
option(HEonGPU_BUILD_BENCHMARKS "Build HEonGPU Benchmarks" OFF)

message(STATUS "HEonGPU_BUILD_TESTS: ${HEonGPU_BUILD_TESTS}")
message(STATUS "HEonGPU_BUILD_EXAMPLES: ${HEonGPU_BUILD_EXAMPLES}")
message(STATUS "HEonGPU_BUILD_BENCHMARKS: ${HEonGPU_BUILD_BENCHMARKS}")

#########
# Tests #
#########

if(HEonGPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

############
# Examples #
############

if(HEonGPU_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

##############
# BENCHMARKS #
##############

if(HEonGPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

############################################################################
# Uninstall Target
############################################################################

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling HEonGPU..."
    )
endif()
