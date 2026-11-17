# Copyright 2024 Alişah Özcan
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# GPU-NTT, GPU-FFT and RNGonGPU are git submodules. Configuring does not fetch
# them, so it needs no network and does not change the source tree, except that
# a HIP build applies the patches below to the submodule checkouts.
foreach(_heongpu_submodule IN ITEMS GPU-FFT GPU-NTT RNGonGPU)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_heongpu_submodule}/CMakeLists.txt")
        message(FATAL_ERROR
            "thirdparty/${_heongpu_submodule} is missing. Clone HEonGPU with "
            "'git clone --recursive', or run "
            "'git submodule update --init --recursive' in the HEonGPU "
            "source directory.")
    endif()
endforeach()

# The AMD GPU support of GPU-NTT, GPU-FFT and RNGonGPU cannot live in their
# checked-out trees, since they are separate repositories pinned here as
# submodules. It is carried as patches applied on top of the pinned commits,
# only for HIP builds; a patch already in place is left alone, so that
# reconfiguring works.
if(USE_HIP)
    find_package(Git REQUIRED)
    foreach(_heongpu_submodule IN ITEMS GPU-NTT GPU-FFT RNGonGPU)
        set(_heongpu_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_heongpu_submodule}.patch")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${_heongpu_patch}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_heongpu_submodule}
            RESULT_VARIABLE _heongpu_patch_applied
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _heongpu_patch_applied EQUAL 0)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} apply ${_heongpu_patch}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_heongpu_submodule}
                RESULT_VARIABLE _heongpu_patch_result
                ERROR_VARIABLE _heongpu_patch_error)
            if(NOT _heongpu_patch_result EQUAL 0)
                message(FATAL_ERROR
                    "Applying ${_heongpu_patch} failed:\n${_heongpu_patch_error}")
            endif()
        endif()
    endforeach()
endif()

include(FetchContent)

if(USE_HIP AND MSVC)
    # The fetched memory manager and the logging library it pulls in are built
    # shared by default, and neither one works that way with an MSVC-style
    # toolchain: their export macros expand to the ELF visibility attribute
    # under __GNUC__ and to nothing otherwise, so the DLLs export no symbols,
    # and the logger hides its spdlog symbols with the GNU-only linker option
    # --exclude-libs, whose operand an MSVC-style link driver reads as an input
    # file name. Building them static sidesteps both, and keeps their code in
    # the executables instead of in DLLs the loader would have to find.
    # See https://github.com/AMD-Ecosystem/rocmds-logger/pull/2 for the linker
    # option; this stays useful after it lands, for the missing exports.
    # Keep this a normal variable: under CMP0077 it makes the memory manager's
    # option(BUILD_SHARED_LIBS ... ON) do nothing, so nothing reaches the cache
    # and the setting stays scoped to this directory. Without it that option()
    # writes BUILD_SHARED_LIBS=ON to the cache and the sibling test/ scope
    # inherits it, which is what turns GoogleTest into a DLL. A cache entry
    # here would instead override a user's own -DBUILD_SHARED_LIBS and outlive
    # a later reconfigure with USE_HIP=OFF.
    set(BUILD_SHARED_LIBS OFF)
endif()

# RMM (RAPIDS Memory Manager)
# On HIP builds the same interface comes from hipMM, the ROCm-DS port of RMM,
# which keeps RMM's target name, rmm:: namespace, and header layout. hipMM has
# its own version numbers.
if(USE_HIP)
    set(_heongpu_rmm_version 5.0)
else()
    set(_heongpu_rmm_version 26.08)
endif()

option(HEonGPU_USE_SYSTEM_RMM
    "Use an installed RMM 26.08 and fail if there is none, instead of downloading and building RMM"
    OFF)
option(HEonGPU_INSTALL_RMM
    "Install the RMM that HEonGPU builds from source together with HEonGPU (the installed HEonGPU needs an RMM 26.08)"
    ON)

if(HEonGPU_USE_SYSTEM_RMM)
    find_package(rmm ${_heongpu_rmm_version} QUIET)
    if(NOT rmm_FOUND)
        message(FATAL_ERROR
            "HEonGPU_USE_SYSTEM_RMM is ON, but no installed RMM ${_heongpu_rmm_version} was "
            "found. Pass the prefix of an RMM ${_heongpu_rmm_version} installation with "
            "CMAKE_PREFIX_PATH (or its config directory with rmm_DIR), or set "
            "HEonGPU_USE_SYSTEM_RMM=OFF to download and build RMM.")
    endif()
else()
    # RMM is downloaded and built. It is not looked up first: an RMM that an
    # earlier HEonGPU install put into the install prefix would be found when
    # reconfiguring, and HEonGPU would then silently stop building and
    # installing its own. Use HEonGPU_USE_SYSTEM_RMM for an installed RMM.
    set(_heongpu_rmm_options "")
    if(NOT HEonGPU_INSTALL_RMM)
        # Also leaves out the install rules of RMM and of its dependencies.
        list(APPEND _heongpu_rmm_options EXCLUDE_FROM_ALL)
    endif()
    if(NOT USE_HIP)
        FetchContent_Declare(
            rmm
            GIT_REPOSITORY https://github.com/rapidsai/rmm
            GIT_TAG        v26.08.00
            SOURCE_SUBDIR  cpp
            ${_heongpu_rmm_options}
        )
    else()
        FetchContent_Declare(
            rmm
            GIT_REPOSITORY https://github.com/AMD-Ecosystem/hipMM
            # hipMM 5.0.0 (release/26.10 line), pinned by commit: hipMM
            # publishes no release tags yet and the branch moves.
            GIT_TAG        dccfd94fc2d513abf3f9f62867e330433f9086e0
            SOURCE_SUBDIR  cpp
            ${_heongpu_rmm_options}
        )
    endif()

    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

    # On HIP, hipMM brings the ROCm counterparts of CCCL itself.
    if(NOT USE_HIP)
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
    endif()

    # RMM looks up its other dependencies with CPM, which prefers installed
    # packages. When reconfiguring, it would find the copies that an earlier
    # HEonGPU install put into the install prefix, and HEonGPU would stop
    # building and installing its own. Build them from source as well.
    foreach(_rmm_pkg IN ITEMS rapids_logger spdlog nvtx3)
        set(CPM_DOWNLOAD_${_rmm_pkg} ON)
        unset(${_rmm_pkg}_DIR CACHE)
    endforeach()

    if(USE_HIP)
        # hipMM needs libhipcxx 3.4 (the CCCL port), newer than the one ROCm
        # ships, and the build scripts it uses download it from a repository
        # that does not exist. The first declaration of a dependency is the one
        # used, so declare the public one here, pinned by commit (the tip of
        # its release/26.10 branch, which hipMM's scripts ask for).
        FetchContent_Declare(
            libhipcxx
            GIT_REPOSITORY https://github.com/ROCm/libhipcxx
            GIT_TAG        90b6a2b5768fecb243d30fb3c0f34223031bd355
        )
    endif()

    FetchContent_MakeAvailable(rmm)
    if(USE_HIP)
        # hipMM expects a HIP-capable compiler for its own sources (its hip
        # dependency injects -x hip into their compile flags, which a plain C++
        # compiler rejects). Compiling them as HIP sources satisfies that the
        # same way this port compiles its own GPU sources.
        get_target_property(_rmm_source_dir rmm SOURCE_DIR)
        get_target_property(_rmm_sources rmm SOURCES)
        list(TRANSFORM _rmm_sources PREPEND "${_rmm_source_dir}/")
        set_source_files_properties(${_rmm_sources} TARGET_DIRECTORY rmm
                                    PROPERTIES LANGUAGE HIP)
    endif()
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
        string(REPLACE "." "\\." _heongpu_rmm_version_regex "${_heongpu_rmm_version}")
        if(EXISTS "${_heongpu_prefix_rmm}" AND
           NOT _heongpu_prefix_rmm_version MATCHES "^${_heongpu_rmm_version_regex}\\.")
            message(WARNING
                "RMM ${_heongpu_prefix_rmm_version} is installed at "
                "${CMAKE_INSTALL_PREFIX}. Installing HEonGPU there replaces it "
                "with the RMM ${_heongpu_rmm_version} built for HEonGPU. To keep it, install "
                "HEonGPU to another prefix, provide an installed RMM ${_heongpu_rmm_version} with "
                "CMAKE_PREFIX_PATH, or set HEonGPU_INSTALL_RMM=OFF and provide "
                "RMM ${_heongpu_rmm_version} when using the installed HEonGPU.")
        endif()
    else()
        message(STATUS "RMM is not installed with HEonGPU (HEonGPU_INSTALL_RMM=OFF)")
    endif()
endif()

# The submodules include <cuda_runtime.h> and <curand_kernel.h> by name. Putting
# hip_compat/ ahead of them on the include path resolves those to HIP shims and
# leaves the submodule sources' include lines untouched.
if(USE_HIP)
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat/ DESTINATION include/hip_compat)
endif()

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

