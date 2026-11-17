// HIP compatibility shim for curand_mtgp32dc_p_11213.h
//
// hipRAND has no equivalent of this header. It supplies the precomputed MTGP32
// parameter set, which the code paths reached on HIP builds never use, so an
// empty header is enough to satisfy the include.
#pragma once

#if !defined(USE_HIP) && !defined(__HIP_PLATFORM_AMD__)
#include_next <curand_mtgp32dc_p_11213.h>
#endif
