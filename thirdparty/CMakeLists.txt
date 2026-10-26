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

option(HEonGPU_USE_SYSTEM_RMM
    "Use an installed RMM 26.08 and fail if there is none, instead of downloading and building RMM"
    OFF)
option(HEonGPU_INSTALL_RMM
    "Install the RMM that HEonGPU builds from source together with HEonGPU (the installed HEonGPU needs an RMM 26.08)"
    ON)

if(HEonGPU_USE_SYSTEM_RMM)
    find_package(rmm 26.08 QUIET)
    if(NOT rmm_FOUND)
        message(FATAL_ERROR
            "HEonGPU_USE_SYSTEM_RMM is ON, but no installed RMM 26.08 was "
            "found. Pass the prefix of an RMM 26.08 installation with "
            "CMAKE_PREFIX_PATH (or its config directory with rmm_DIR), or set "
            "HEonGPU_USE_SYSTEM_RMM=OFF to download and build RMM.")
    endif()
else()
    # An installed RMM 26.08 is used when find_package finds one; otherwise
    # RMM is downloaded and built.
    set(_heongpu_rmm_options "")
    if(NOT HEonGPU_INSTALL_RMM)
        # Also leaves out the install rules of RMM and of its dependencies.
        list(APPEND _heongpu_rmm_options EXCLUDE_FROM_ALL)
    endif()
    FetchContent_Declare(
        rmm
        GIT_REPOSITORY https://github.com/rapidsai/rmm
        GIT_TAG        v26.08.00
        SOURCE_SUBDIR  cpp
        ${_heongpu_rmm_options}
        FIND_PACKAGE_ARGS 26.08
    )

    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

    # RMM is written against the CCCL version it pins (3.4.x for RMM 26.08).
    # The CCCL shipped with the CUDA Toolkit can be older (CUDA 12.x ships CCCL
    # 2.x) or newer than that, and rapids-cmake would pick it up whenever it
    # satisfies the minimum version. Always fetch the pinned CCCL.
    set(CPM_DOWNLOAD_CCCL ON CACHE BOOL "Fetch the CCCL version pinned by RMM instead of using the CUDA Toolkit one")
    if(CPM_DOWNLOAD_CCCL)
        # Drop package locations cached by an earlier configure (e.g. pointing
        # at the CUDA Toolkit's CCCL); otherwise the fetched Thrust would be
        # paired with a mismatched CUB/libcudacxx from the toolkit.
        foreach(_cccl_pkg IN ITEMS CCCL Thrust CUB libcudacxx)
            unset(${_cccl_pkg}_DIR CACHE)
        endforeach()
    endif()

    FetchContent_MakeAvailable(rmm)
endif()

if(rmm_FOUND)
    message(STATUS "Using the installed RMM ${rmm_VERSION}: ${rmm_DIR}")
else()
    message(STATUS "Using RMM built from source")
    if(HEonGPU_INSTALL_RMM)
        # Installing HEonGPU also installs this RMM, which replaces another
        # RMM version installed at the same prefix.
        set(_heongpu_prefix_rmm
            "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/rmm/rmm-config-version.cmake")
        if(EXISTS "${_heongpu_prefix_rmm}")
            file(STRINGS "${_heongpu_prefix_rmm}" _heongpu_prefix_rmm_version
                REGEX "^set\\(PACKAGE_VERSION \"")
            string(REGEX REPLACE ".*\"([^\"]*)\".*" "\\1"
                _heongpu_prefix_rmm_version "${_heongpu_prefix_rmm_version}")
        endif()
        if(EXISTS "${_heongpu_prefix_rmm}" AND
           NOT _heongpu_prefix_rmm_version MATCHES "^26\\.08\\.")
            message(WARNING
                "RMM ${_heongpu_prefix_rmm_version} is installed at "
                "${CMAKE_INSTALL_PREFIX}. Installing HEonGPU there replaces it "
                "with the RMM 26.08 built for HEonGPU. To keep it, install "
                "HEonGPU to another prefix, provide an installed RMM 26.08 with "
                "CMAKE_PREFIX_PATH, or set HEonGPU_INSTALL_RMM=OFF and provide "
                "RMM 26.08 when using the installed HEonGPU.")
        endif()
    else()
        message(STATUS "RMM is not installed with HEonGPU (HEonGPU_INSTALL_RMM=OFF)")
    endif()
endif()

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

