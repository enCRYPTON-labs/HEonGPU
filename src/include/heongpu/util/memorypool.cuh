// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_MEMORY_POOL_H
#define HEONGPU_MEMORY_POOL_H

#include <mutex>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_set>
#include <sys/sysinfo.h>

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include <heongpu/kernel/defines.h>
#include <heongpu/util/util.cuh>

#include <thrust/host_vector.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

namespace heongpu
{
    struct MemoryPoolConfig
    {
        // Fractions accept either 0.0-1.0 (ratio) or 0-100 (percentage).
        std::optional<float> initial_device_fraction;
        std::optional<float> max_device_fraction;
        std::optional<size_t> initial_device_bytes;
        std::optional<size_t> max_device_bytes;

        std::optional<float> initial_host_fraction;
        std::optional<float> max_host_fraction;
        std::optional<size_t> initial_host_bytes;
        std::optional<size_t> max_host_bytes;

        bool use_memory_pool = true;

        // Also make HEonGPU's device memory resource the process-wide RMM
        // default (rmm::mr::set_current_device_resource), so that the other
        // RMM allocations of the process use HEonGPU's pool. HEonGPU does not
        // need this itself. The previous default is restored when a later
        // context turns it off and at exit.
        bool set_as_rmm_default = false;

        static MemoryPoolConfig Defaults();
    };

    class MemoryPool
    {
        using DeviceResource = rmm::mr::cuda_memory_resource;
        using DevicePoolResource = rmm::mr::pool_memory_resource;
        using DeviceStatsAdaptor = rmm::mr::statistics_resource_adaptor;

        using HostResource = rmm::mr::pinned_host_memory_resource;
        using HostPoolResource = rmm::mr::pool_memory_resource;
        using HostStatsAdaptor = rmm::mr::statistics_resource_adaptor;

      public:
        static MemoryPool& instance();

        void initialize();
        void initialize(const MemoryPoolConfig& config);
        // for device
        void use_memory_pool(bool use);
        // See MemoryPoolConfig::set_as_rmm_default.
        void set_as_rmm_default(bool enable);

        // for device
        void* allocate(size_t size, cudaStream_t stream = cudaStreamDefault);
        void deallocate(void* ptr, size_t size,
                        cudaStream_t stream = cudaStreamDefault);

        rmm::device_async_resource_ref get_device_resource() const;
        HostStatsAdaptor* get_host_resource() const;

        void* host_allocate(size_t size);
        void host_deallocate(void* ptr, size_t size);

        void print_memory_pool_status() const;
        size_t get_current_device_pool_memory_usage() const;
        size_t get_free_device_pool_memory() const;

        size_t get_current_host_pool_memory_usage() const;
        size_t get_free_host_pool_memory() const;

        ~MemoryPool();

      private:
        MemoryPool();
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        void clean_pool();
        void ensure_base_resources();
        rmm::device_async_resource_ref device_resource();
        size_t get_host_avaliable_memory() const;
        size_t get_decive_avaliable_memory() const;
        size_t roundup_256(size_t size) const;

        static std::shared_ptr<HostResource> host_base_;
        static std::shared_ptr<HostPoolResource> host_pool_;
        static std::shared_ptr<HostStatsAdaptor> host_stats_adaptor_;

        static std::shared_ptr<DeviceResource> device_base_;
        static std::shared_ptr<DevicePoolResource> device_pool_;
        static std::shared_ptr<DeviceStatsAdaptor> device_stats_adaptor_;
        static bool initialized_;
        // Set by use_memory_pool(); selects the resource for new allocations.
        static bool use_pool_;
        // Raw and host allocations served by the pools, so that they are
        // returned to them even after use_memory_pool(false).
        static std::unordered_set<void*> pooled_device_allocations_;
        static std::unordered_set<void*> pooled_host_allocations_;
        // The RMM default that set_as_rmm_default(true) replaced, restored by
        // set_as_rmm_default(false) and at exit.
        static std::optional<
            cuda::mr::any_resource<cuda::mr::device_accessible>>
            previous_rmm_default_;
        static std::mutex mutex_;
    };

    template <typename T> struct rmm_pinned_allocator
    {
        using value_type = T;
        using HostResource = rmm::mr::pinned_host_memory_resource;
        using HostPoolResource = rmm::mr::pool_memory_resource;

        rmm_pinned_allocator() = default;

        T* allocate(std::size_t n)
        {
            return static_cast<T*>(
                MemoryPool::instance().host_allocate(n * sizeof(T)));
        }

        void deallocate(T* p, std::size_t n)
        {
            MemoryPool::instance().host_deallocate(p, n * sizeof(T));
        }

        bool operator==(const rmm_pinned_allocator& other) const
        {
            return true;
        }
        bool operator!=(const rmm_pinned_allocator& other) const
        {
            return !(*this == other);
        }
    };

} // namespace heongpu

#endif // HEONGPU_MEMORY_POOL_H
