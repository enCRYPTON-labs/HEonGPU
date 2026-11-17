// HIP compatibility shim for cuda_runtime.h
// This header redirects to hip_runtime.h when building with HIP.

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

// CUDA runtime -> HIP runtime symbol aliases
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

#define cudaStream_t                hipStream_t
#define cudaStreamDefault           hipStreamDefault
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamNonBlocking       hipStreamNonBlocking
#define cudaStreamPerThread         hipStreamPerThread
#define cudaStreamWaitEvent         hipStreamWaitEvent
#define cudaLaunchHostFunc          hipLaunchHostFunc

#define cudaEvent_t                 hipEvent_t
#define cudaEventCreate             hipEventCreate
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventDestroy            hipEventDestroy
#define cudaEventElapsedTime        hipEventElapsedTime
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDisableTiming      hipEventDisableTiming

#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess
#define cudaGetErrorString          hipGetErrorString
#define cudaGetLastError            hipGetLastError
#define cudaPeekAtLastError         hipPeekAtLastError
#define cudaDeviceSynchronize       hipDeviceSynchronize

#define cudaGetDevice               hipGetDevice
#define cudaSetDevice               hipSetDevice
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceProp              hipDeviceProp_t

// hipPointerAttribute_t exposes .type and has no .memoryType member, which is
// the shape CUDA 10.0+ has. Claiming that version steers version-guarded code
// to the .type branch.
#ifndef CUDART_VERSION
#define CUDART_VERSION 10000
#endif

// Pointer attributes
#define cudaPointerAttributes       hipPointerAttribute_t
#define cudaPointerGetAttributes    hipPointerGetAttributes

// Memory types
#define cudaMemoryTypeDevice        hipMemoryTypeDevice
#define cudaMemoryTypeHost          hipMemoryTypeHost
#define cudaMemoryTypeUnregistered  hipMemoryTypeUnregistered
#define cudaMemoryTypeManaged       hipMemoryTypeManaged

// Device attributes
#define cudaDeviceGetAttribute      hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported

// Async memory allocation
#define cudaMallocAsync             hipMallocAsync
#define cudaFreeAsync               hipFreeAsync
#define cudaMallocFromPoolAsync     hipMallocFromPoolAsync
#define cudaMemPool_t               hipMemPool_t
#define cudaMemPoolProps            hipMemPoolProps
#define cudaMemPoolCreate           hipMemPoolCreate
#define cudaMemPoolDestroy          hipMemPoolDestroy
#define cudaMemPoolSetAttribute     hipMemPoolSetAttribute
#define cudaMemPoolGetAttribute     hipMemPoolGetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#define cudaMemLocationTypeDevice   hipMemLocationTypeDevice

// Memory info
#define cudaMemGetInfo              hipMemGetInfo

// Function attributes and occupancy
#define cudaFuncSetAttribute        hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

// Device limits
#define cudaDeviceSetLimit          hipDeviceSetLimit
#define cudaDeviceGetLimit          hipDeviceGetLimit
#define cudaLimitStackSize          hipLimitStackSize

#else
#include_next <cuda_runtime.h>
#endif
