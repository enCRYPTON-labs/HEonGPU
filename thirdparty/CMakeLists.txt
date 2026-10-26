# Copyright 2024 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# GPU-NTT, GPU-FFT and RNGonGPU are git submodules. Configuring does not fetch
# them, so it needs no network and does not change the source tree.
foreach(_heongpu_submodule IN ITEMS GPU-FFT GPU-NTT RNGonGPU)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_heongpu_submodule}/CMakeLists.txt")
        message(FATAL_ERROR
            "thirdparty/${_heongpu_submodule} is missing. Clone HEonGPU with "
            "'git clone --recursive', or run "
            "'git submodule update --init --recursive' in the HEonGPU "
            "source directory.")
    endif()
endforeach()

include(FetchContent)

# An installed RMM 26.08 is used when find_package finds one; otherwise RMM is
# downloaded and built.
FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        v26.08.00
    SOURCE_SUBDIR  cpp
    FIND_PACKAGE_ARGS 26.08
)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

# RMM is written against the CCCL version it pins (3.4.x for RMM 26.08).
# The CCCL shipped with the CUDA Toolkit can be older (CUDA 12.x ships CCCL 2.x)
# or newer than that, and rapids-cmake would pick it up whenever it satisfies
# the minimum version. Always fetch the pinned CCCL.
set(CPM_DOWNLOAD_CCCL ON CACHE BOOL "Fetch the CCCL version pinned by RMM instead of using the CUDA Toolkit one")
if(CPM_DOWNLOAD_CCCL)
    # Drop package locations cached by an earlier configure (e.g. pointing at the
    # CUDA Toolkit's CCCL); otherwise the fetched Thrust would be paired with a
    # mismatched CUB/libcudacxx from the toolkit.
    foreach(_cccl_pkg IN ITEMS CCCL Thrust CUB libcudacxx)
        unset(${_cccl_pkg}_DIR CACHE)
    endforeach()
endif()

FetchContent_MakeAvailable(rmm)
if(rmm_FOUND)
    message(STATUS "Using the installed RMM ${rmm_VERSION}: ${rmm_DIR}")
else()
    message(STATUS "Using RMM built from source")
endif()

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

