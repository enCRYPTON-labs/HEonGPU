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
#include <string>
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
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

namespace heongpu
{
    /**
     * @enum memory_topology
     * @brief Where the memory of the GPU is.
     *
     * - `discrete`: the GPU has its own memory, separate from the memory of
     *   the host.
     * - `unified`: the GPU and the host share one physical memory, so the
     *   device and host pools take memory from the same place.
     */
    enum class memory_topology : std::uint8_t
    {
        discrete = 0x1,
        unified = 0x2
    };

    /**
     * @enum memory_mode
     * @brief How HEonGPU allocates its device memory.
     *
     * - `automatic`: `unified` on a GPU that shares the memory of the host
     *   and lets the host use managed memory while the GPU runs, `copy`
     *   otherwise.
     * - `copy`: device memory that only the GPU uses (cudaMalloc), with
     *   copies between host and device memory.
     * - `unified`: managed memory (cudaMallocManaged) that the host and the
     *   GPU both use. Requires concurrent managed access; on a GPU with its
     *   own memory it works, more slowly, and is meant for tests.
     *
     * The environment variable HEONGPU_MEMORY_MODE (automatic, copy or
     * unified) overrides MemoryPoolConfig::memory_mode.
     */
    enum class memory_mode : std::uint8_t
    {
        automatic = 0x1,
        copy = 0x2,
        unified = 0x3
    };

    namespace detail
    {
        // The device attributes and memory sizes that tell the memory
        // topology of a GPU.
        struct DeviceMemoryAttributes
        {
            // cudaDevAttrIntegrated.
            int integrated = 0;
            // cudaDevAttrPageableMemoryAccessUsesHostPageTables.
            int uses_host_page_tables = 0;
            // cudaDevAttrConcurrentManagedAccess.
            int concurrent_managed_access = 0;
            // Total memory of the GPU (cudaMemGetInfo) and of the host
            // (sysinfo), in bytes.
            size_t device_memory = 0;
            size_t host_memory = 0;
        };

        // Queries the attributes of device.
        DeviceMemoryAttributes query_device_memory_attributes(int device);

        // An integrated GPU shares the memory of the host. A GPU that uses
        // the page tables of the host and reports a total memory within 10%
        // of the host memory is taken as sharing it too; a GPU with its own
        // memory that is coherent with the host, but of another size, is
        // discrete.
        memory_topology
        classify_memory_topology(const DeviceMemoryAttributes& attributes);

        // The memory the host can still give to the process, in bytes:
        // MemAvailable of /proc/meminfo, which counts the page cache that the
        // kernel can reclaim, lowered to what the memory cgroups of the
        // process (v1 or v2, and their parents) still allow, as in a
        // container with a memory limit. Returns 0 when MemAvailable cannot
        // be read. root is prepended to the paths, for tests.
        size_t read_available_host_memory(const std::string& root = "");

        // The memory mode to use for the requested one on a GPU with the
        // given topology and attributes: copy or unified. Throws
        // std::invalid_argument when unified is requested and the GPU has no
        // concurrent managed access.
        memory_mode
        select_memory_mode(memory_mode requested, memory_topology topology,
                           const DeviceMemoryAttributes& attributes);

        // The memory mode named by value (automatic, copy or unified), or
        // fallback when value is null or empty. Throws std::invalid_argument
        // for another value.
        memory_mode memory_mode_from_environment(const char* value,
                                                 memory_mode fallback);
    } // namespace detail

    struct MemoryPoolConfig
    {
        // Fractions accept either 0.0-1.0 (ratio) or 0-100 (percentage).
        //
        // On a GPU with its own memory, the device fractions are of the free
        // device memory and the host fractions of the available host memory.
        // On a GPU that shares the memory of the host (memory_topology::
        // unified), both pools take from one budget, 80% of the available
        // host memory, and the fractions are of that budget: fractions whose
        // sum is above 100% are scaled down to it, and a pool given in bytes
        // leaves the rest of the budget to the other one. Sizes in bytes that
        // do not fit in the budget together are rejected.
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

        // HEonGPU's kernels need a per-thread stack of at least 2048 bytes.
        // When the CUDA limit (cudaLimitStackSize) is lower, generate() raises
        // it for the whole process; a higher limit is kept. Turn this off to
        // manage the limit in the application; generate() then throws
        // StateError instead of raising a limit that is too low.
        bool raise_kernel_stack_size = true;

        // See memory_mode; HEONGPU_MEMORY_MODE overrides it.
        heongpu::memory_mode memory_mode = heongpu::memory_mode::automatic;

        // When the device pool takes the memory of the host (a GPU that
        // shares it, or memory_mode::unified), it returns the memory it does
        // not use to the system at synchronizations, beyond this many bytes
        // that it keeps for new allocations. 1 GB by default. Such a pool
        // also keeps at most this much of its initial size.
        std::optional<size_t> device_release_threshold_bytes;

        static MemoryPoolConfig Defaults();
    };

    namespace detail
    {
        // How the device pool keeps memory.
        // - growing: it keeps all the memory it took until it is destroyed.
        // - releasing: a CUDA stream-ordered pool that returns the memory it
        //   does not use beyond its release threshold at synchronizations.
        enum class device_pool_kind : std::uint8_t
        {
            growing = 0x1,
            releasing = 0x2
        };

        // releasing when the device pool takes the memory of the host (a GPU
        // with memory_topology::unified, or memory_mode::unified) and the
        // GPU supports stream-ordered pools; growing otherwise.
        device_pool_kind select_device_pool_kind(memory_topology topology,
                                                 memory_mode mode,
                                                 bool pools_supported);

        // For tests: the kind of device pool for the initializations that
        // follow, instead of the selected one; std::nullopt to select again.
        void override_device_pool_kind(std::optional<device_pool_kind> kind);
        std::optional<device_pool_kind> overridden_device_pool_kind();

        // Sizes of the pools in bytes.
        struct PoolBudget
        {
            size_t initial_device = 0;
            size_t max_device = 0;
            size_t initial_host = 0;
            size_t max_host = 0;
        };

        // The pool sizes of config for a GPU with the given topology, with
        // device_available bytes of free device memory and host_available
        // bytes of available host memory. Throws std::invalid_argument for
        // sizes that cannot be met.
        PoolBudget resolve_pool_budget(const MemoryPoolConfig& config,
                                       memory_topology topology,
                                       size_t device_available,
                                       size_t host_available);
    } // namespace detail

    class MemoryPool
    {
        using DeviceResource = rmm::mr::cuda_memory_resource;
        using ManagedResource = rmm::mr::managed_memory_resource;
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
        // Destroys the pools, so that the next initialize() creates new ones
        // with a new configuration, and restores the RMM default. Throws
        // StateError, without changing anything, while pool memory is in use.
        // Used by heongpu::shutdown().
        void release();

        // for device
        void* allocate(size_t size, cudaStream_t stream = cudaStreamDefault);
        void deallocate(void* ptr, size_t size,
                        cudaStream_t stream = cudaStreamDefault);

        rmm::device_async_resource_ref get_device_resource() const;
        HostStatsAdaptor* get_host_resource() const;

        void* host_allocate(size_t size);
        void host_deallocate(void* ptr, size_t size);

        // Writes the pool sizes and usage to os, std::cout by default.
        void print_memory_pool_status(std::ostream& os = std::cout) const;
        size_t get_current_device_pool_memory_usage() const;
        size_t get_free_device_pool_memory() const;

        size_t get_current_host_pool_memory_usage() const;
        size_t get_free_host_pool_memory() const;

        // The memory topology of the GPU, detected by initialize().
        // discrete before the first initialization.
        memory_topology topology() const;

        // The memory mode chosen by initialize(): copy or unified. copy
        // before the first initialization.
        memory_mode mode() const;

        // The kind of the device pool chosen by initialize().
        detail::device_pool_kind device_pool_kind() const;

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
        static std::shared_ptr<ManagedResource> managed_base_;
        static std::shared_ptr<DevicePoolResource> device_pool_;
        // The releasing device pool: a stream-ordered pool of device memory,
        // or one of managed memory that HEonGPU creates, below an adaptor
        // that limits it to the max size.
        static std::shared_ptr<rmm::mr::cuda_async_memory_resource>
            async_device_pool_;
        static cudaMemPool_t managed_pool_handle_;
        static std::shared_ptr<rmm::mr::cuda_async_view_memory_resource>
            managed_pool_;
        static std::shared_ptr<rmm::mr::limiting_resource_adaptor>
            limited_device_pool_;
        static detail::device_pool_kind device_pool_kind_;
        static size_t release_threshold_;
        // Bytes the device pool holds: its size, or the memory reserved by
        // the stream-ordered pool. The caller holds mutex_.
        size_t device_pool_size() const;
        // Creates the releasing device pool; false when the GPU or the CUDA
        // version does not support it. The caller holds mutex_.
        bool create_releasing_device_pool(int device,
                                          const detail::PoolBudget& budget);
        void destroy_device_pool();
        static std::shared_ptr<DeviceStatsAdaptor> device_stats_adaptor_;
        static bool initialized_;
        static detail::DeviceMemoryAttributes attributes_;
        static memory_topology topology_;
        static memory_mode mode_;
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
