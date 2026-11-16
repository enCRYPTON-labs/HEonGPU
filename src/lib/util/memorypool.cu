// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/memorypool.cuh>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>

namespace heongpu
{
    std::shared_ptr<MemoryPool::HostResource> MemoryPool::host_base_ = nullptr;
    std::shared_ptr<MemoryPool::HostPoolResource> MemoryPool::host_pool_ =
        nullptr;
    std::shared_ptr<MemoryPool::HostStatsAdaptor>
        MemoryPool::host_stats_adaptor_ = nullptr;

    std::shared_ptr<MemoryPool::DeviceResource> MemoryPool::device_base_ =
        nullptr;
    std::shared_ptr<MemoryPool::DevicePoolResource> MemoryPool::device_pool_ =
        nullptr;
    std::shared_ptr<MemoryPool::DeviceStatsAdaptor>
        MemoryPool::device_stats_adaptor_ = nullptr;
    bool MemoryPool::initialized_ = false;
    detail::DeviceMemoryAttributes MemoryPool::attributes_;
    memory_topology MemoryPool::topology_ = memory_topology::discrete;
    bool MemoryPool::use_pool_ = true;
    std::unordered_set<void*> MemoryPool::pooled_device_allocations_;
    std::unordered_set<void*> MemoryPool::pooled_host_allocations_;
    std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
        MemoryPool::previous_rmm_default_;
    std::mutex MemoryPool::mutex_;

    namespace detail
    {
        DeviceMemoryAttributes query_device_memory_attributes(int device)
        {
            cudaDeviceProp properties;
            HEONGPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            DeviceMemoryAttributes attributes;
            attributes.integrated = properties.integrated;
            attributes.uses_host_page_tables =
                properties.pageableMemoryAccessUsesHostPageTables;
            attributes.concurrent_managed_access =
                properties.concurrentManagedAccess;
            attributes.device_memory = properties.totalGlobalMem;
            struct sysinfo memory_info;
            if (sysinfo(&memory_info) == 0)
            {
                attributes.host_memory =
                    size_t(memory_info.totalram) * memory_info.mem_unit;
            }
            return attributes;
        }

        namespace
        {
            // The first value of a cgroup file such as memory.max; false for
            // "max" (no limit) or an unreadable file.
            bool read_cgroup_value(const std::string& path, size_t& value)
            {
                std::ifstream file(path);
                std::string token;
                if (!(file >> token) || token == "max")
                {
                    return false;
                }
                try
                {
                    value = std::stoull(token);
                }
                catch (const std::exception&)
                {
                    return false;
                }
                return true;
            }

            // The value of key in a file of "key value" lines, 0 when it is
            // missing.
            size_t read_keyed_value(const std::string& path,
                                    const std::string& key)
            {
                std::ifstream file(path);
                std::string name;
                size_t value = 0;
                while (file >> name >> value)
                {
                    if (name == key)
                    {
                        return value;
                    }
                    file.ignore(std::numeric_limits<std::streamsize>::max(),
                                '\n');
                }
                return 0;
            }

            // Lowers available to what the cgroup at directory and its
            // parents up to base still allow. The page cache that the kernel
            // can reclaim (inactive_key of memory.stat) is not counted as
            // used.
            void limit_by_cgroup(size_t& available, const std::string& base,
                                 std::string directory,
                                 const std::string& limit_file,
                                 const std::string& usage_file,
                                 const std::string& inactive_key)
            {
                while (true)
                {
                    size_t limit = 0;
                    size_t usage = 0;
                    if (read_cgroup_value(directory + "/" + limit_file,
                                          limit) &&
                        read_cgroup_value(directory + "/" + usage_file, usage))
                    {
                        const size_t inactive = std::min(
                            usage, read_keyed_value(directory + "/memory.stat",
                                                    inactive_key));
                        const size_t used = usage - inactive;
                        available = std::min(available,
                                             limit > used ? limit - used : 0);
                    }
                    if (directory.size() <= base.size())
                    {
                        break;
                    }
                    directory = directory.substr(0, directory.rfind('/'));
                }
            }
        } // namespace

        size_t read_available_host_memory(const std::string& root)
        {
            std::ifstream meminfo(root + "/proc/meminfo");
            std::string line;
            size_t available = 0;
            bool found = false;
            while (std::getline(meminfo, line))
            {
                std::istringstream fields(line);
                std::string name;
                size_t kilobytes = 0;
                if (fields >> name >> kilobytes && name == "MemAvailable:")
                {
                    available = kilobytes * 1024;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return 0;
            }

            // Lines of /proc/self/cgroup are "id:controllers:path"; cgroup
            // v2 has id 0 and no controllers.
            std::ifstream cgroups(root + "/proc/self/cgroup");
            while (std::getline(cgroups, line))
            {
                const size_t first = line.find(':');
                const size_t second = line.find(':', first + 1);
                if (first == std::string::npos || second == std::string::npos)
                {
                    continue;
                }
                const std::string id = line.substr(0, first);
                const std::string controllers =
                    "," + line.substr(first + 1, second - first - 1) + ",";
                std::string path = line.substr(second + 1);
                while (!path.empty() && path.back() == '/')
                {
                    path.pop_back();
                }
                if (id == "0" && controllers == ",,")
                {
                    const std::string base = root + "/sys/fs/cgroup";
                    limit_by_cgroup(available, base, base + path, "memory.max",
                                    "memory.current", "inactive_file");
                }
                else if (controllers.find(",memory,") != std::string::npos)
                {
                    const std::string base = root + "/sys/fs/cgroup/memory";
                    limit_by_cgroup(
                        available, base, base + path, "memory.limit_in_bytes",
                        "memory.usage_in_bytes", "total_inactive_file");
                }
            }
            return available;
        }

        memory_topology
        classify_memory_topology(const DeviceMemoryAttributes& attributes)
        {
            if (attributes.integrated != 0)
            {
                return memory_topology::unified;
            }
            if (attributes.uses_host_page_tables != 0 &&
                attributes.host_memory != 0)
            {
                const size_t difference =
                    attributes.device_memory > attributes.host_memory
                        ? attributes.device_memory - attributes.host_memory
                        : attributes.host_memory - attributes.device_memory;
                if (difference < attributes.host_memory / 10)
                {
                    return memory_topology::unified;
                }
            }
            return memory_topology::discrete;
        }
    } // namespace detail

    namespace detail
    {
        namespace
        {
            size_t roundup_256(size_t size)
            {
                return ((size + 255) / 256) * 256;
            }

            float normalize_fraction(float value, const char* label)
            {
                if (value <= 0.0f)
                {
                    throw std::invalid_argument(std::string(label) +
                                                " must be > 0");
                }
                if (value > 1.0f)
                {
                    if (value > 100.0f)
                    {
                        throw std::invalid_argument(
                            std::string(label) +
                            " must be in (0,1] or (0,100]");
                    }
                    return value / 100.0f;
                }
                return value;
            }

            void check_bytes(const std::optional<size_t>& bytes,
                             const char* label)
            {
                if (bytes.has_value() && *bytes == 0)
                {
                    throw std::invalid_argument(std::string(label) +
                                                " bytes must be > 0");
                }
            }

            void check_resolved(size_t resolved, size_t total_memory,
                                const char* label)
            {
                if (resolved == 0)
                {
                    throw std::invalid_argument(std::string(label) +
                                                " resolved to 0 bytes");
                }
                if (total_memory != 0 && resolved > total_memory)
                {
                    throw std::invalid_argument(
                        std::string(label) +
                        " exceeds available memory at initialization");
                }
            }

            // A fraction is taken in float, as before the shared budget, or
            // in double when precise.
            size_t resolve_pool_size(const std::optional<size_t>& bytes,
                                     const std::optional<float>& fraction,
                                     size_t total_memory, size_t default_bytes,
                                     float default_fraction, const char* label,
                                     bool precise = false)
            {
                auto part = [&](float f)
                {
                    return precise
                               ? static_cast<size_t>(double(total_memory) * f)
                               : static_cast<size_t>(total_memory * f);
                };
                check_bytes(bytes, label);
                size_t resolved = 0;
                if (bytes.has_value())
                {
                    resolved = *bytes;
                }
                else if (fraction.has_value())
                {
                    resolved = part(normalize_fraction(*fraction, label));
                }
                else if (default_bytes != 0)
                {
                    resolved = default_bytes;
                }
                else
                {
                    resolved =
                        part(normalize_fraction(default_fraction, label));
                }
                check_resolved(resolved, total_memory, label);
                return roundup_256(resolved);
            }

            size_t resolve_initial_host(const MemoryPoolConfig& config,
                                        size_t total_memory,
                                        bool precise = false)
            {
                if (!config.initial_host_bytes.has_value() &&
                    !config.initial_host_fraction.has_value())
                {
                    return roundup_256(static_cast<size_t>(104857600));
                }
                return resolve_pool_size(
                    config.initial_host_bytes, config.initial_host_fraction,
                    total_memory, 0, initial_host_memorypool_size,
                    "host initial pool size", precise);
            }

            void check_initial(size_t initial, size_t max, const char* pool)
            {
                if (max < initial)
                {
                    throw std::invalid_argument(std::string(pool) +
                                                " max pool size must be >= " +
                                                pool + " initial pool size");
                }
            }

            PoolBudget resolve_discrete_budget(const MemoryPoolConfig& config,
                                               size_t device_available,
                                               size_t host_available)
            {
                PoolBudget budget;
                budget.initial_host =
                    resolve_initial_host(config, host_available);
                budget.max_host = resolve_pool_size(
                    config.max_host_bytes, config.max_host_fraction,
                    host_available, 0, max_host_memorypool_size,
                    "host max pool size");
                check_initial(budget.initial_host, budget.max_host, "host");

                budget.initial_device = resolve_pool_size(
                    config.initial_device_bytes, config.initial_device_fraction,
                    device_available, 0, initial_device_memorypool_size,
                    "device initial pool size");
                budget.max_device = resolve_pool_size(
                    config.max_device_bytes, config.max_device_fraction,
                    device_available, 0, max_device_memorypool_size,
                    "device max pool size");
                check_initial(budget.initial_device, budget.max_device,
                              "device");
                return budget;
            }

            // Both pools share total; see MemoryPoolConfig.
            PoolBudget resolve_unified_budget(const MemoryPoolConfig& config,
                                              size_t host_available)
            {
                // In double: a float is off by kilobytes at these sizes.
                const size_t total = static_cast<size_t>(
                    double(host_available) * unified_memorypool_budget);
                constexpr size_t min_device = 64ULL << 20;
                constexpr size_t min_host = 256ULL << 20;

                check_bytes(config.max_device_bytes, "device max pool size");
                check_bytes(config.max_host_bytes, "host max pool size");
                const float device_fraction =
                    config.max_device_fraction.has_value()
                        ? normalize_fraction(*config.max_device_fraction,
                                             "device max pool size")
                        : unified_max_device_memorypool_size;
                const float host_fraction =
                    config.max_host_fraction.has_value()
                        ? normalize_fraction(*config.max_host_fraction,
                                             "host max pool size")
                        : unified_max_host_memorypool_size;

                PoolBudget budget;
                if (config.max_device_bytes.has_value() &&
                    config.max_host_bytes.has_value())
                {
                    budget.max_device = *config.max_device_bytes;
                    budget.max_host = *config.max_host_bytes;
                    if (budget.max_device + budget.max_host > total)
                    {
                        throw std::invalid_argument(
                            "device and host max pool sizes together exceed "
                            "the memory available at initialization, which "
                            "both pools share on this GPU");
                    }
                }
                else if (config.max_device_bytes.has_value())
                {
                    budget.max_device = *config.max_device_bytes;
                    check_resolved(budget.max_device, total,
                                   "device max pool size");
                    budget.max_host = std::max(
                        min_host, std::min(static_cast<size_t>(double(total) *
                                                               host_fraction),
                                           total - budget.max_device));
                }
                else if (config.max_host_bytes.has_value())
                {
                    budget.max_host = *config.max_host_bytes;
                    check_resolved(budget.max_host, total,
                                   "host max pool size");
                    budget.max_device =
                        std::max(min_device,
                                 std::min(static_cast<size_t>(double(total) *
                                                              device_fraction),
                                          total - budget.max_host));
                }
                else
                {
                    const double sum = double(device_fraction) + host_fraction;
                    const double scale = sum > 1.0 ? 1.0 / sum : 1.0;
                    budget.max_device = static_cast<size_t>(
                        double(total) * device_fraction * scale);
                    budget.max_host =
                        std::min(static_cast<size_t>(double(total) *
                                                     host_fraction * scale),
                                 total - budget.max_device);
                }
                check_resolved(budget.max_device, total,
                               "device max pool size");
                check_resolved(budget.max_host, total, "host max pool size");
                budget.max_device = roundup_256(budget.max_device);
                budget.max_host = roundup_256(budget.max_host);

                // Initial sizes from a fraction or the defaults stay within
                // the max size; one given in bytes must fit.
                budget.initial_host = resolve_initial_host(config, total, true);
                if (!config.initial_host_bytes.has_value())
                {
                    budget.initial_host =
                        std::min(budget.initial_host, budget.max_host);
                }
                check_initial(budget.initial_host, budget.max_host, "host");

                budget.initial_device = resolve_pool_size(
                    config.initial_device_bytes, config.initial_device_fraction,
                    total, unified_initial_device_memorypool_bytes, 0.0f,
                    "device initial pool size", true);
                if (!config.initial_device_bytes.has_value())
                {
                    budget.initial_device =
                        std::min(budget.initial_device, budget.max_device);
                }
                check_initial(budget.initial_device, budget.max_device,
                              "device");
                return budget;
            }
        } // namespace

        PoolBudget resolve_pool_budget(const MemoryPoolConfig& config,
                                       memory_topology topology,
                                       size_t device_available,
                                       size_t host_available)
        {
            if (topology == memory_topology::unified)
            {
                return resolve_unified_budget(config, host_available);
            }
            return resolve_discrete_budget(config, device_available,
                                           host_available);
        }
    } // namespace detail

    MemoryPoolConfig MemoryPoolConfig::Defaults()
    {
        MemoryPoolConfig config;
        config.initial_device_fraction = initial_device_memorypool_size;
        config.max_device_fraction = max_device_memorypool_size;
        config.max_host_fraction = max_host_memorypool_size;
        return config;
    }

    MemoryPool& MemoryPool::instance()
    {
        // Initialize CUDA runtime before the singleton is constructed so that
        // teardown order stays correct at process exit.
        cudaFree(nullptr);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        static MemoryPool instance;
        return instance;
    }

    void MemoryPool::ensure_base_resources()
    {
        if (!host_base_)
        {
            host_base_ = std::make_shared<HostResource>();
        }
        if (!device_base_)
        {
            device_base_ = std::make_shared<DeviceResource>();
        }
    }

    size_t MemoryPool::get_host_avaliable_memory() const
    {
        const size_t available = detail::read_available_host_memory();
        if (available != 0)
        {
            return available;
        }
        struct sysinfo memInfo;
        sysinfo(&memInfo);
        size_t free_memory = memInfo.freeram * memInfo.mem_unit;
        return free_memory;
    }

    size_t MemoryPool::get_decive_avaliable_memory() const
    {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        return free_mem; // total_mem
    }

    size_t MemoryPool::roundup_256(size_t size) const
    {
        return ((size + 255) / 256) * 256;
    }

    void MemoryPool::initialize()
    {
        initialize(MemoryPoolConfig::Defaults());
    }

    void MemoryPool::initialize(const MemoryPoolConfig& config)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        // The pools are created by the first initialization that requests
        // them, even if an earlier one ran without a pool.
        if (!initialized_ || (config.use_memory_pool && !device_pool_))
        {
            ensure_base_resources();
            int device = 0;
            HEONGPU_CUDA_CHECK(cudaGetDevice(&device));
            attributes_ = detail::query_device_memory_attributes(device);
            topology_ = detail::classify_memory_topology(attributes_);
            const size_t host_available = get_host_avaliable_memory();
            const size_t device_available = get_decive_avaliable_memory();

            if (config.use_memory_pool)
            {
                const detail::PoolBudget budget = detail::resolve_pool_budget(
                    config, topology_, device_available, host_available);

                host_pool_ = std::make_shared<HostPoolResource>(
                    *host_base_, budget.initial_host, budget.max_host);
                host_stats_adaptor_ =
                    std::make_shared<HostStatsAdaptor>(*host_pool_);

                size_t initial_device_pool_size = budget.initial_device;
                const size_t max_device_pool_size = budget.max_device;

                // An initial size derived from the free memory can become too
                // large if another process allocates in the meantime. Retry
                // with smaller sizes; the pool grows on demand up to its max.
                constexpr size_t min_initial_device_pool_size = 64ULL << 20;
                const bool shrinkable =
                    !config.initial_device_bytes.has_value();
                while (true)
                {
                    if (shrinkable)
                    {
                        // Skip sizes that no longer fit instead of attempting
                        // them, which avoids failed allocations and RMM
                        // error logs.
                        constexpr size_t headroom = 64ULL << 20;
                        // A GPU that shares the memory of the host takes
                        // the pool from the available host memory.
                        const size_t free_memory =
                            topology_ == memory_topology::unified
                                ? get_host_avaliable_memory()
                                : get_decive_avaliable_memory();
                        while (initial_device_pool_size >
                                   min_initial_device_pool_size &&
                               initial_device_pool_size + headroom >
                                   free_memory)
                        {
                            initial_device_pool_size =
                                roundup_256(initial_device_pool_size / 2);
                        }
                    }
                    try
                    {
                        device_pool_ = std::make_shared<DevicePoolResource>(
                            *device_base_, initial_device_pool_size,
                            max_device_pool_size);
                        break;
                    }
                    catch (const rmm::out_of_memory&)
                    {
                        if (!shrinkable || initial_device_pool_size <=
                                               min_initial_device_pool_size)
                        {
                            throw;
                        }
                        initial_device_pool_size =
                            roundup_256(initial_device_pool_size / 2);
                    }
                }
                device_stats_adaptor_ =
                    std::make_shared<DeviceStatsAdaptor>(*device_pool_);
            }

            initialized_ = true;
        }
    }

    void MemoryPool::use_memory_pool(bool use)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        use_pool_ = use;
        const rmm::device_async_resource_ref resource = device_resource();
        if (previous_rmm_default_)
        {
            // Keep the RMM default on the resource HEonGPU now uses.
            rmm::mr::set_current_device_resource(resource);
        }
    }

    void MemoryPool::set_as_rmm_default(bool enable)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (enable)
        {
            auto previous =
                rmm::mr::set_current_device_resource(device_resource());
            // Keep the default from before HEonGPU first replaced it.
            if (!previous_rmm_default_)
            {
                previous_rmm_default_ = std::move(previous);
            }
        }
        else if (previous_rmm_default_)
        {
            rmm::mr::set_current_device_resource(
                std::move(*previous_rmm_default_));
            previous_rmm_default_.reset();
        }
    }

    void MemoryPool::release()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        const size_t device_used =
            device_stats_adaptor_
                ? static_cast<size_t>(
                      device_stats_adaptor_->get_bytes_counter().value)
                : 0;
        const size_t host_used =
            host_stats_adaptor_
                ? static_cast<size_t>(
                      host_stats_adaptor_->get_bytes_counter().value)
                : 0;
        if (device_used != 0 || host_used != 0)
        {
            throw StateError(error_message(
                "HEonGPU can not be shut down while its memory pool is in "
                "use (",
                device_used, " bytes of device memory and ", host_used,
                " bytes of host memory). Destroy all HEonGPU objects "
                "(contexts, keys, plaintexts, ciphertexts, encoders, "
                "encryptors, decryptors and operators) first."));
        }
        if (previous_rmm_default_)
        {
            rmm::mr::set_current_device_resource(
                std::move(*previous_rmm_default_));
            previous_rmm_default_.reset();
        }
        // The base resources are kept: memory that was allocated without the
        // pool can still be returned to them.
        device_stats_adaptor_.reset();
        device_pool_.reset();
        host_stats_adaptor_.reset();
        host_pool_.reset();
        pooled_device_allocations_.clear();
        pooled_host_allocations_.clear();
        initialized_ = false;
        use_pool_ = true;
    }

    rmm::device_async_resource_ref MemoryPool::device_resource()
    {
        if (use_pool_ && device_stats_adaptor_)
        {
            return rmm::device_async_resource_ref{*device_stats_adaptor_};
        }
        ensure_base_resources();
        return rmm::device_async_resource_ref{*device_base_};
    }

    void* MemoryPool::allocate(size_t size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (use_pool_ && device_stats_adaptor_)
        {
            void* ptr = device_stats_adaptor_->allocate(
                stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
            pooled_device_allocations_.insert(ptr);
            return ptr;
        }
        ensure_base_resources();
        return device_base_->allocate(stream, size,
                                      rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    void MemoryPool::deallocate(void* ptr, size_t size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (pooled_device_allocations_.erase(ptr) != 0)
        {
            device_stats_adaptor_->deallocate(stream, ptr, size,
                                              rmm::CUDA_ALLOCATION_ALIGNMENT);
            return;
        }
        ensure_base_resources();
        device_base_->deallocate(stream, ptr, size,
                                 rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    rmm::device_async_resource_ref MemoryPool::get_device_resource() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return const_cast<MemoryPool*>(this)->device_resource();
    }

    MemoryPool::HostStatsAdaptor* MemoryPool::get_host_resource() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return host_stats_adaptor_.get();
    }

    void* MemoryPool::host_allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (use_pool_ && host_stats_adaptor_)
        {
            void* ptr = host_stats_adaptor_->allocate(
                cuda::stream_ref{cudaStream_t{nullptr}}, size,
                rmm::CUDA_ALLOCATION_ALIGNMENT);
            pooled_host_allocations_.insert(ptr);
            return ptr;
        }
        ensure_base_resources();
        return host_base_->allocate(cuda::stream_ref{cudaStream_t{nullptr}},
                                    size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    void MemoryPool::host_deallocate(void* ptr, size_t size)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (pooled_host_allocations_.erase(ptr) != 0)
        {
            host_stats_adaptor_->deallocate(
                cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size,
                rmm::CUDA_ALLOCATION_ALIGNMENT);
            return;
        }
        ensure_base_resources();
        host_base_->deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr,
                               size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    void MemoryPool::print_memory_pool_status(std::ostream& os) const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (device_stats_adaptor_ && device_pool_ && host_stats_adaptor_ &&
            host_pool_)
        {
            auto device_status = device_stats_adaptor_->get_bytes_counter();
            const auto device_total = device_pool_->pool_size();
            const auto device_used = device_status.value;
            const auto device_free = device_total - device_used;

            auto host_status = host_stats_adaptor_->get_bytes_counter();
            const auto host_total = host_pool_->pool_size();
            const auto host_used = host_status.value;
            const auto host_free = host_total - host_used;

            os << "[HEonGPU] Memory Pool Status" << std::endl;
            os << "  Topology    : "
               << (topology_ == memory_topology::unified ? "unified"
                                                         : "discrete")
               << " (integrated " << attributes_.integrated
               << ", host page tables " << attributes_.uses_host_page_tables
               << ", concurrent managed access "
               << attributes_.concurrent_managed_access << ")" << std::endl;
            os << "  Device Pool:" << std::endl;
            os << "    Total     : " << device_total << " bytes" << std::endl;
            os << "    Used      : " << device_used << " bytes" << std::endl;
            os << "    Free      : " << device_free << " bytes" << std::endl;
            os << "  Host Pool:" << std::endl;
            os << "    Total     : " << host_total << " bytes" << std::endl;
            os << "    Used      : " << host_used << " bytes" << std::endl;
            os << "    Free      : " << host_free << " bytes" << std::endl;
        }
        else
        {
            os << "[HEonGPU] Memory pool is not initialized or is disabled."
               << std::endl;
        }
    }

    size_t MemoryPool::get_current_device_pool_memory_usage() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!device_stats_adaptor_)
        {
            return 0;
        }
        auto device_status = device_stats_adaptor_->get_bytes_counter();
        return device_status.value;
    }

    size_t MemoryPool::get_free_device_pool_memory() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!device_stats_adaptor_ || !device_pool_)
        {
            return 0;
        }
        auto device_status = device_stats_adaptor_->get_bytes_counter();
        return device_pool_->pool_size() - device_status.value;
    }

    size_t MemoryPool::get_current_host_pool_memory_usage() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!host_stats_adaptor_)
        {
            return 0;
        }
        auto host_status = host_stats_adaptor_->get_bytes_counter();
        return host_status.value;
    }

    size_t MemoryPool::get_free_host_pool_memory() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!host_stats_adaptor_ || !host_pool_)
        {
            return 0;
        }
        auto host_status = host_stats_adaptor_->get_bytes_counter();
        return host_pool_->pool_size() - host_status.value;
    }

    memory_topology MemoryPool::topology() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return topology_;
    }

    MemoryPool::~MemoryPool()
    {
        clean_pool();
    }

    MemoryPool::MemoryPool() = default;

    void MemoryPool::clean_pool()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (initialized_ || host_base_ || device_base_ || host_pool_ ||
            device_pool_)
        {
            // Restore the RMM default before destroying the resources it may
            // refer to; a default set by the application stays as it is.
            if (previous_rmm_default_)
            {
                rmm::mr::set_current_device_resource(
                    std::move(*previous_rmm_default_));
                previous_rmm_default_.reset();
            }
            host_stats_adaptor_.reset();
            host_pool_.reset();
            host_base_.reset();
            device_stats_adaptor_.reset();
            device_pool_.reset();
            device_base_.reset();
            pooled_device_allocations_.clear();
            pooled_host_allocations_.clear();
            use_pool_ = true;
            initialized_ = false;
        }
    }

} // namespace heongpu
