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
    GIT_TAG        branch-25.08
    SOURCE_SUBDIR  cpp
)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

# RMM 25.08 is written against the CCCL version it pins (3.0.x) and uses
# cuda::mr::async_resource_ref, which newer CCCL releases removed. CUDA 13.x
# ships such a newer CCCL (e.g. 3.4 in CUDA 13.4), and rapids-cmake would pick
# it up since it satisfies the minimum version. Always fetch the pinned CCCL.
# (CUDA 12.x ships CCCL 2.x, which is too old anyway, so it was already fetched.)
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

