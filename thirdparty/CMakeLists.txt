# Copyright 2024 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE BUILD_SCRIPT_RESULT
    OUTPUT_VARIABLE BUILD_SCRIPT_OUTPUT
    ERROR_VARIABLE BUILD_SCRIPT_ERROR
)

if(NOT BUILD_SCRIPT_RESULT EQUAL 0)
    message(FATAL_ERROR "Thirdparty build script failed:\n${BUILD_SCRIPT_ERROR}")
endif()

message(STATUS "Thirdparty build completed successfully")

include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        v26.08.00
    SOURCE_SUBDIR  cpp
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

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

