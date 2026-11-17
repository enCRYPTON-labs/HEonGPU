// HIP compatibility shim for curand_mtgp32_host.h
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hiprand/hiprand_mtgp32_host.h>
#else
#include_next <curand_mtgp32_host.h>
#endif
