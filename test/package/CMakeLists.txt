# Copyright 2024-2026 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# A project that uses an installed HEonGPU, to check the installed package.
# Configure it with CMAKE_PREFIX_PATH set to the HEonGPU install prefix.

cmake_minimum_required(VERSION 4.0)
project(heongpu_package_test LANGUAGES CXX)

# A static HEonGPU leaves the CUDA device link to the project that uses it; a
# shared HEonGPU does not need CUDA in that project.
option(HEONGPU_PACKAGE_TEST_CUDA "Build as a CUDA project, as a static HEonGPU requires" ON)
if(HEONGPU_PACKAGE_TEST_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(HEonGPU REQUIRED)

add_executable(package_test main.cpp)
target_link_libraries(package_test PRIVATE HEonGPU::heongpu)
if(HEONGPU_PACKAGE_TEST_CUDA)
    target_link_libraries(package_test PRIVATE CUDA::cudart)
    set_target_properties(package_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()
