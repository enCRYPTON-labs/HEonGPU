// HIP compatibility shim for curand_kernel.h
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hiprand/hiprand_kernel.h>

// cuRAND device API -> hipRAND device API
#define curandState                 hiprandState
#define curandState_t               hiprandState_t
#define curandStateXORWOW           hiprandStateXORWOW
#define curandStateXORWOW_t         hiprandStateXORWOW_t
#define curandStateMRG32k3a         hiprandStateMRG32k3a
#define curandStateMRG32k3a_t       hiprandStateMRG32k3a_t
#define curandStatePhilox4_32_10    hiprandStatePhilox4_32_10
#define curandStatePhilox4_32_10_t  hiprandStatePhilox4_32_10_t
#define curand_init                 hiprand_init
#define curand                      hiprand
#define curand_uniform              hiprand_uniform
#define curand_uniform_double       hiprand_uniform_double
#define curand_normal               hiprand_normal
#define curand_normal_double        hiprand_normal_double

#else
#include_next <curand_kernel.h>
#endif
