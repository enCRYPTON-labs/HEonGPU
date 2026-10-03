# Locate CCCL when it is installed with the CUDA Toolkit.

set(_CCCL_HINT_PREFIXES
    ${CCCL_ROOT}
    ${CCCL_DIR}
    $ENV{CCCL_ROOT}
    $ENV{CCCL_DIR}
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_ROOT}
    ${CUDAToolkit_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
    /usr/local/cuda)
list(REMOVE_DUPLICATES _CCCL_HINT_PREFIXES)

set(_CCCL_CONFIG_PATHS)
foreach(_prefix IN LISTS _CCCL_HINT_PREFIXES)
    if(_prefix AND EXISTS "${_prefix}")
        # The hint may already be the config directory itself (e.g. rmm's
        # installed config sets CCCL_ROOT to <prefix>/lib/rapids/cmake/cccl).
        list(APPEND _CCCL_CONFIG_PATHS
            "${_prefix}"
            "${_prefix}/lib64/cmake/cccl"
            "${_prefix}/lib/cmake/cccl"
            "${_prefix}/share/cmake/cccl")
    endif()
endforeach()

find_file(CCCL_CONFIG_FILE
    NAMES cccl-config.cmake CCCLConfig.cmake
    PATHS ${_CCCL_CONFIG_PATHS}
    NO_DEFAULT_PATH)

if(NOT CCCL_CONFIG_FILE)
    find_file(CCCL_CONFIG_FILE
        NAMES cccl-config.cmake CCCLConfig.cmake
        PATH_SUFFIXES lib64/cmake/cccl lib/cmake/cccl share/cmake/cccl)
endif()

if(CCCL_CONFIG_FILE)
    get_filename_component(CCCL_DIR "${CCCL_CONFIG_FILE}" DIRECTORY)
    include("${CCCL_CONFIG_FILE}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CCCL DEFAULT_MSG CCCL_CONFIG_FILE)

mark_as_advanced(CCCL_CONFIG_FILE CCCL_DIR)

unset(_CCCL_HINT_PREFIXES)
unset(_CCCL_CONFIG_PATHS)
