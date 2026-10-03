# Locate Thrust library and set THRUST_INCLUDE_DIR
#
# CUDA <= 12.x ships Thrust under <cuda>/include/thrust, while CUDA >= 13.0
# moved it (together with CUB and libcu++) to <cuda>/include/cccl/thrust.
# The CUDAToolkit paths are searched first so the Thrust that matches the
# selected nvcc is found even when several toolkits are installed.

find_path(THRUST_INCLUDE_DIR
  NAMES thrust/version.h
  HINTS
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_LIBRARY_ROOT}/include
    ${CUDAToolkit_TARGET_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_SDK_ROOT_DIR}
    /usr/local/cuda/include
    /usr/include/cuda
    /usr/local/include
  PATH_SUFFIXES
    cccl
)

if (THRUST_INCLUDE_DIR)
  list(REMOVE_DUPLICATES THRUST_INCLUDE_DIR)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust REQUIRED_VARS THRUST_INCLUDE_DIR)

if(Thrust_FOUND AND NOT TARGET Thrust)
  add_library(Thrust INTERFACE)
  target_include_directories(Thrust INTERFACE ${THRUST_INCLUDE_DIR})
endif()

mark_as_advanced(THRUST_INCLUDE_DIR)
