// HIP compatibility shim for device_launch_parameters.h
// On HIP, hip_runtime.h provides all the necessary launch parameters.
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// HIP does not need a separate device_launch_parameters.h
// All necessary definitions are in hip_runtime.h
#include <hip/hip_runtime.h>
#else
#include_next <device_launch_parameters.h>
#endif
