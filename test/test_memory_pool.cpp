// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace
{
    heongpu::HEContext<heongpu::Scheme::CKKS> make_context(bool use_memory_pool)
    {
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});

        heongpu::MemoryPoolConfig pool_config;
        pool_config.initial_device_bytes = 256ULL * 1024 * 1024;
        pool_config.max_device_bytes = 1024ULL * 1024 * 1024;
        pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
        pool_config.max_host_bytes = 256ULL * 1024 * 1024;
        pool_config.use_memory_pool = use_memory_pool;
        context->generate(pool_config);
        return context;
    }

    constexpr size_t count = 1 << 16;
    constexpr size_t bytes = count * sizeof(double);
} // namespace

TEST(HEonGPU, MemoryPool_Tracks_Host_And_Device_Usage)
{
    auto context = make_context(true);

    auto& pool = heongpu::MemoryPool::instance();

    const size_t host_before = pool.get_current_host_pool_memory_usage();
    const size_t device_before = pool.get_current_device_pool_memory_usage();

    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();

        EXPECT_GE(pool.get_current_host_pool_memory_usage(),
                  host_before + bytes);
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_before + bytes);
    }

    cudaDeviceSynchronize();

    EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_before);
    EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_before);
}

TEST(HEonGPU, MemoryPool_Respects_Use_Memory_Pool_Flag)
{
    auto& pool = heongpu::MemoryPool::instance();

    auto pooled_context = make_context(true);
    const size_t host_base = pool.get_current_host_pool_memory_usage();
    const size_t device_base = pool.get_current_device_pool_memory_usage();

    auto pooled_host =
        std::make_unique<heongpu::HostVector<double>>(count, 0.0);
    auto pooled_device = std::make_unique<heongpu::DeviceVector<double>>(count);
    void* pooled_raw = pool.allocate(bytes);
    cudaDeviceSynchronize();

    const size_t host_pooled = pool.get_current_host_pool_memory_usage();
    const size_t device_pooled = pool.get_current_device_pool_memory_usage();
    EXPECT_GE(host_pooled, host_base + bytes);
    EXPECT_GE(device_pooled, device_base + 2 * bytes);

    // A context created with use_memory_pool = false disables the pool.
    auto unpooled_context = make_context(false);
    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        void* raw = pool.allocate(bytes);
        cudaDeviceSynchronize();

        EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_pooled);
        EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_pooled);

        pool.deallocate(raw, bytes);
    }
    cudaDeviceSynchronize();

    // Buffers allocated from the pool are returned to it while it is disabled.
    pooled_host.reset();
    pooled_device.reset();
    pool.deallocate(pooled_raw, bytes);
    cudaDeviceSynchronize();

    EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_base);
    EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_base);

    // Re-enabling the pool routes new allocations through it again.
    auto repooled_context = make_context(true);
    const size_t host_repooled = pool.get_current_host_pool_memory_usage();
    const size_t device_repooled = pool.get_current_device_pool_memory_usage();
    {
        heongpu::HostVector<double> host_vector(count, 0.0);
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();

        EXPECT_GE(pool.get_current_host_pool_memory_usage(),
                  host_repooled + bytes);
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_repooled + bytes);
    }
    cudaDeviceSynchronize();
}

TEST(HEonGPU, MemoryPool_Can_Be_Enabled_After_Unpooled_Context)
{
    // The first context after shutdown() disables the pool.
    auto& pool = heongpu::MemoryPool::instance();
    // Start from a process without a memory pool.
    heongpu::shutdown();
    auto unpooled_context = make_context(false);
    EXPECT_EQ(pool.get_free_device_pool_memory(), 0u);

    auto pooled_context = make_context(true);
    EXPECT_NE(pool.get_free_device_pool_memory(), 0u);
    const size_t device_before = pool.get_current_device_pool_memory_usage();
    {
        heongpu::DeviceVector<double> device_vector(count);
        cudaDeviceSynchronize();
        EXPECT_GE(pool.get_current_device_pool_memory_usage(),
                  device_before + bytes);
    }
    cudaDeviceSynchronize();
}

TEST(HEonGPU, MemoryPool_Context_Generation_With_Little_Free_Device_Memory)
{
    // Simulates another process holding most of the GPU before the first
    // context of this process is generated with the default pool settings.
    auto& pool = heongpu::MemoryPool::instance();
    // Start from a process without a memory pool.
    heongpu::shutdown();
    size_t free_memory = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_memory, &total_memory), cudaSuccess);
    const size_t leave_free = 1536ULL * 1024 * 1024;
    if (free_memory < 2 * leave_free)
    {
        GTEST_SKIP() << "Not enough free device memory for this test.";
    }
    void* blocker = nullptr;
    ASSERT_EQ(cudaMalloc(&blocker, free_memory - leave_free), cudaSuccess);

    {
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});
        ASSERT_NO_THROW(context->generate());
        EXPECT_NE(pool.get_free_device_pool_memory(), 0u);

        heongpu::DeviceVector<double> device_vector(count);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    EXPECT_EQ(cudaFree(blocker), cudaSuccess);
}

TEST(HEonGPU, MemoryPool_Shrinks_Initial_Size_That_Does_Not_Fit)
{
    auto& pool = heongpu::MemoryPool::instance();
    // Start from a process without a memory pool.
    heongpu::shutdown();
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30}, {40});

    // All of the free memory cannot be reserved for the pool at once.
    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 1.0f;
    pool_config.max_device_fraction = 1.0f;
    pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
    pool_config.max_host_bytes = 256ULL * 1024 * 1024;
    // Sizes that do not fit are skipped instead of being attempted, so RMM
    // does not log allocation failures.
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(context->generate(pool_config));
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_EQ(log.find("maximum pool size exceeded"), std::string::npos) << log;

    size_t free_memory = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_memory, &total_memory), cudaSuccess);
    EXPECT_NE(pool.get_free_device_pool_memory(), 0u);
    EXPECT_LT(pool.get_free_device_pool_memory(), total_memory);

    heongpu::DeviceVector<double> device_vector(count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(HEonGPU, MemoryPool_Initial_Device_Fraction_Can_Be_Chosen_By_User)
{
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    if (heongpu::detail::classify_memory_topology(
            heongpu::detail::query_device_memory_attributes(device)) ==
        heongpu::memory_topology::unified)
    {
        GTEST_SKIP() << "The fraction is of the shared budget on this GPU; "
                        "MemoryPool_Budget_Of_A_Unified_GPU checks it.";
    }
    auto& pool = heongpu::MemoryPool::instance();
    // Start from a process without a memory pool.
    heongpu::shutdown();
    size_t free_before = 0;
    size_t total_memory = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total_memory), cudaSuccess);

    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(4096);
    context->set_coeff_modulus_bit_sizes({40, 30}, {40});

    // Reserve more than the default up front.
    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 80.0f;
    pool_config.initial_host_bytes = 64ULL * 1024 * 1024;
    pool_config.max_host_bytes = 256ULL * 1024 * 1024;
    context->generate(pool_config);

    if (pool.device_pool_kind() == heongpu::detail::device_pool_kind::releasing)
    {
        GTEST_SKIP() << "A releasing device pool keeps at most its release "
                        "threshold of the initial size.";
    }
    const double reserved =
        static_cast<double>(pool.get_free_device_pool_memory() +
                            pool.get_current_device_pool_memory_usage());
    EXPECT_GT(reserved, 0.7 * static_cast<double>(free_before));
    EXPECT_LT(reserved, 0.85 * static_cast<double>(free_before));
}

TEST(HEonGPU, MemoryPool_Classifies_The_Memory_Topology)
{
    using heongpu::memory_topology;
    using heongpu::detail::classify_memory_topology;
    using Attributes = heongpu::detail::DeviceMemoryAttributes;
    constexpr size_t GB = 1ULL << 30;

    // A GPU with its own memory, with and without HMM.
    EXPECT_EQ(classify_memory_topology(Attributes{0, 0, 1, 32 * GB, 128 * GB}),
              memory_topology::discrete);
    EXPECT_EQ(classify_memory_topology(Attributes{0, 0, 1, 128 * GB, 128 * GB}),
              memory_topology::discrete);
    // A GPU with its own memory that is coherent with the host and uses its
    // page tables, but has another size.
    EXPECT_EQ(classify_memory_topology(Attributes{0, 1, 1, 96 * GB, 480 * GB}),
              memory_topology::discrete);
    EXPECT_EQ(classify_memory_topology(Attributes{0, 1, 1, 100 * GB, 128 * GB}),
              memory_topology::discrete);
    // An integrated GPU.
    EXPECT_EQ(classify_memory_topology(Attributes{1, 0, 0, 8 * GB, 8 * GB}),
              memory_topology::unified);
    EXPECT_EQ(classify_memory_topology(Attributes{1, 1, 1, 120 * GB, 128 * GB}),
              memory_topology::unified);
    // A GPU that uses the page tables of the host and reports about the
    // memory of the host.
    EXPECT_EQ(classify_memory_topology(Attributes{0, 1, 1, 120 * GB, 125 * GB}),
              memory_topology::unified);
    // The same when the memory of the host is unknown.
    EXPECT_EQ(classify_memory_topology(Attributes{0, 1, 1, 120 * GB, 0}),
              memory_topology::discrete);
}

TEST(HEonGPU, MemoryPool_Reports_The_Memory_Topology)
{
    auto context = make_context(true);
    auto& pool = heongpu::MemoryPool::instance();

    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    const heongpu::memory_topology expected =
        heongpu::detail::classify_memory_topology(
            heongpu::detail::query_device_memory_attributes(device));
    EXPECT_EQ(pool.topology(), expected);

    std::ostringstream status;
    pool.print_memory_pool_status(status);
    const std::string line = expected == heongpu::memory_topology::unified
                                 ? "Topology    : unified"
                                 : "Topology    : discrete";
    EXPECT_NE(status.str().find(line), std::string::npos) << status.str();
}

namespace
{
    // A directory with the files that read_available_host_memory reads.
    class FakeRoot
    {
      public:
        explicit FakeRoot(const std::string& name)
            : path_(std::filesystem::path(::testing::TempDir()) /
                    ("heongpu_memory_" + name))
        {
            std::filesystem::remove_all(path_);
        }
        ~FakeRoot() { std::filesystem::remove_all(path_); }

        void write(const std::string& file, const std::string& content)
        {
            const std::filesystem::path full = path_ / file;
            std::filesystem::create_directories(full.parent_path());
            std::ofstream(full) << content;
        }

        std::string path() const { return path_.string(); }

      private:
        std::filesystem::path path_;
    };

    constexpr size_t MB = 1ULL << 20;

    std::string meminfo(size_t available)
    {
        return "MemTotal:       67108864 kB\nMemFree:         1048576 kB\n"
               "MemAvailable:   " +
               std::to_string(available / 1024) + " kB\n";
    }
} // namespace

TEST(HEonGPU, MemoryPool_Reads_The_Available_Host_Memory)
{
    using heongpu::detail::read_available_host_memory;

    {
        FakeRoot root("no_cgroup");
        root.write("proc/meminfo", meminfo(8192 * MB));
        EXPECT_EQ(read_available_host_memory(root.path()), 8192 * MB);
    }
    {
        // The reclaimable page cache of the cgroup is not counted as used.
        FakeRoot root("v2_limit");
        root.write("proc/meminfo", meminfo(8192 * MB));
        root.write("proc/self/cgroup", "0::/\n");
        root.write("sys/fs/cgroup/memory.max", std::to_string(4096 * MB));
        root.write("sys/fs/cgroup/memory.current", std::to_string(3072 * MB));
        root.write("sys/fs/cgroup/memory.stat",
                   "anon 1\ninactive_file " + std::to_string(1024 * MB) + "\n");
        EXPECT_EQ(read_available_host_memory(root.path()), 2048 * MB);
    }
    {
        FakeRoot root("v2_no_limit");
        root.write("proc/meminfo", meminfo(8192 * MB));
        root.write("proc/self/cgroup", "0::/\n");
        root.write("sys/fs/cgroup/memory.max", "max\n");
        root.write("sys/fs/cgroup/memory.current", std::to_string(3072 * MB));
        EXPECT_EQ(read_available_host_memory(root.path()), 8192 * MB);
    }
    {
        // A parent cgroup with a tighter limit.
        FakeRoot root("v2_nested");
        root.write("proc/meminfo", meminfo(8192 * MB));
        root.write("proc/self/cgroup", "0::/a/b\n");
        root.write("sys/fs/cgroup/a/b/memory.max", "max\n");
        root.write("sys/fs/cgroup/a/b/memory.current", "0\n");
        root.write("sys/fs/cgroup/a/memory.max", std::to_string(2048 * MB));
        root.write("sys/fs/cgroup/a/memory.current", std::to_string(512 * MB));
        EXPECT_EQ(read_available_host_memory(root.path()), 1536 * MB);
    }
    {
        FakeRoot root("v1_limit");
        root.write("proc/meminfo", meminfo(8192 * MB));
        root.write("proc/self/cgroup", "12:cpu,cpuacct:/\n4:memory:/job\n");
        root.write("sys/fs/cgroup/memory/job/memory.limit_in_bytes",
                   std::to_string(2048 * MB));
        root.write("sys/fs/cgroup/memory/job/memory.usage_in_bytes",
                   std::to_string(1536 * MB));
        root.write("sys/fs/cgroup/memory/job/memory.stat",
                   "total_inactive_file " + std::to_string(512 * MB) + "\n");
        EXPECT_EQ(read_available_host_memory(root.path()), 1024 * MB);
    }
    {
        FakeRoot root("no_meminfo");
        root.write("proc/self/cgroup", "0::/\n");
        EXPECT_EQ(read_available_host_memory(root.path()), 0u);
    }

#ifndef _WIN32
    // The host of the test.
    const size_t available = read_available_host_memory();
    struct sysinfo info;
    ASSERT_EQ(sysinfo(&info), 0);
    EXPECT_GT(available, 0u);
    EXPECT_LE(available, size_t(info.totalram) * info.mem_unit);
#endif
}

namespace
{
    constexpr size_t GB = 1ULL << 30;

    size_t roundup_256(size_t size)
    {
        return ((size + 255) / 256) * 256;
    }

    // The size the pool gives to a fraction of total, as the pool computes
    // it: in float for a discrete GPU, in double for the shared budget.
    size_t fraction_of(size_t total, float fraction)
    {
        return roundup_256(static_cast<size_t>(total * fraction));
    }

    size_t budget_fraction_of(size_t total, float fraction)
    {
        return roundup_256(static_cast<size_t>(double(total) * fraction));
    }
} // namespace

TEST(HEonGPU, MemoryPool_Budget_Of_A_Discrete_GPU)
{
    using heongpu::detail::resolve_pool_budget;
    const auto discrete = heongpu::memory_topology::discrete;

    heongpu::MemoryPoolConfig config;
    auto budget = resolve_pool_budget(config, discrete, 10 * GB, 20 * GB);
    EXPECT_EQ(budget.initial_device, fraction_of(10 * GB, 0.5f));
    EXPECT_EQ(budget.max_device, fraction_of(10 * GB, 0.95f));
    EXPECT_EQ(budget.initial_host, roundup_256(100ULL << 20));
    EXPECT_EQ(budget.max_host, fraction_of(20 * GB, 0.4f));

    // Each pool has its own memory, so fractions of both can be large.
    config.max_device_fraction = 90.0f;
    config.max_host_fraction = 90.0f;
    budget = resolve_pool_budget(config, discrete, 10 * GB, 20 * GB);
    EXPECT_EQ(budget.max_device, fraction_of(10 * GB, 0.9f));
    EXPECT_EQ(budget.max_host, fraction_of(20 * GB, 0.9f));

    config = heongpu::MemoryPoolConfig();
    config.max_device_bytes = 11 * GB;
    EXPECT_THROW(resolve_pool_budget(config, discrete, 10 * GB, 20 * GB),
                 std::invalid_argument);

    config = heongpu::MemoryPoolConfig();
    config.initial_device_bytes = 2 * GB;
    config.max_device_bytes = 1 * GB;
    EXPECT_THROW(resolve_pool_budget(config, discrete, 10 * GB, 20 * GB),
                 std::invalid_argument);
}

TEST(HEonGPU, MemoryPool_Budget_Of_A_Unified_GPU)
{
    using heongpu::detail::resolve_pool_budget;
    const auto unified = heongpu::memory_topology::unified;
    // 80% of the available host memory; the free device memory, which is
    // the same memory, is not used.
    const size_t host_available = 100 * GB;
    const size_t total = static_cast<size_t>(double(host_available) * 0.8f);
    const size_t device_available = 1 * GB;
    auto expect_within_budget = [&](const heongpu::detail::PoolBudget& budget)
    {
        EXPECT_LE(budget.max_device + budget.max_host, total + 512);
        EXPECT_LE(budget.initial_device, budget.max_device);
        EXPECT_LE(budget.initial_host, budget.max_host);
    };

    heongpu::MemoryPoolConfig config;
    auto budget =
        resolve_pool_budget(config, unified, device_available, host_available);
    expect_within_budget(budget);
    EXPECT_EQ(budget.initial_device, 64ULL << 20);
    EXPECT_EQ(budget.max_device, budget_fraction_of(total, 0.75f));
    EXPECT_EQ(budget.initial_host, roundup_256(100ULL << 20));
    EXPECT_EQ(budget.max_host, budget_fraction_of(total, 0.25f));

    // Fractions above 100% together are scaled down to the budget.
    config.initial_device_fraction = 80.0f;
    config.max_device_fraction = 90.0f;
    config.max_host_fraction = 50.0f;
    budget =
        resolve_pool_budget(config, unified, device_available, host_available);
    expect_within_budget(budget);
    EXPECT_GT(budget.max_device + budget.max_host, total - total / 100);
    EXPECT_NEAR(double(budget.max_device) / budget.max_host, 0.9 / 0.5, 0.01);
    // The initial size of 80% does not fit in the scaled max size.
    EXPECT_EQ(budget.initial_device, budget.max_device);

    // A pool given in bytes leaves the rest to the other one.
    config = heongpu::MemoryPoolConfig();
    config.initial_device_fraction = 1.0f;
    config.max_device_fraction = 1.0f;
    config.initial_host_bytes = 64ULL << 20;
    config.max_host_bytes = 256ULL << 20;
    budget =
        resolve_pool_budget(config, unified, device_available, host_available);
    expect_within_budget(budget);
    // An initial size of 100% of the budget is not rejected for rounding.
    EXPECT_EQ(budget.max_host, 256ULL << 20);
    EXPECT_EQ(budget.max_device, roundup_256(total - (256ULL << 20)));
    EXPECT_EQ(budget.initial_device, budget.max_device);

    // The same with a host memory that a float does not hold exactly.
    for (size_t available : {size_t(100 * GB) + 1, size_t(127 * GB) + 4097,
                             size_t(31 * GB) + 123457})
    {
        EXPECT_NO_THROW(
            resolve_pool_budget(config, unified, device_available, available));
    }

    config = heongpu::MemoryPoolConfig();
    config.max_device_bytes = 70 * GB;
    budget =
        resolve_pool_budget(config, unified, device_available, host_available);
    expect_within_budget(budget);
    EXPECT_EQ(budget.max_device, 70 * GB);
    EXPECT_EQ(budget.max_host, roundup_256(total - 70 * GB));

    // Sizes in bytes that do not fit together, or alone.
    config = heongpu::MemoryPoolConfig();
    config.max_device_bytes = 70 * GB;
    config.max_host_bytes = 20 * GB;
    EXPECT_THROW(
        resolve_pool_budget(config, unified, device_available, host_available),
        std::invalid_argument);
    config = heongpu::MemoryPoolConfig();
    config.max_device_bytes = 90 * GB;
    EXPECT_THROW(
        resolve_pool_budget(config, unified, device_available, host_available),
        std::invalid_argument);
    config = heongpu::MemoryPoolConfig();
    config.initial_device_bytes = 30 * GB;
    config.max_device_bytes = 20 * GB;
    EXPECT_THROW(
        resolve_pool_budget(config, unified, device_available, host_available),
        std::invalid_argument);
}

TEST(HEonGPU, MemoryPool_Selects_The_Memory_Mode)
{
    using heongpu::memory_mode;
    using heongpu::memory_topology;
    using heongpu::detail::select_memory_mode;
    using Attributes = heongpu::detail::DeviceMemoryAttributes;
    const Attributes managed{0, 0, 1, 0, 0};
    const Attributes no_managed{1, 0, 0, 0, 0};

    EXPECT_EQ(select_memory_mode(memory_mode::automatic,
                                 memory_topology::discrete, managed),
              memory_mode::copy);
    EXPECT_EQ(select_memory_mode(memory_mode::automatic,
                                 memory_topology::unified, managed),
              memory_mode::unified);
    // Without concurrent managed access the host can not use managed memory
    // while the GPU runs.
    EXPECT_EQ(select_memory_mode(memory_mode::automatic,
                                 memory_topology::unified, no_managed),
              memory_mode::copy);
    EXPECT_EQ(select_memory_mode(memory_mode::copy, memory_topology::unified,
                                 managed),
              memory_mode::copy);
    EXPECT_EQ(select_memory_mode(memory_mode::unified,
                                 memory_topology::discrete, managed),
              memory_mode::unified);
    EXPECT_THROW(select_memory_mode(memory_mode::unified,
                                    memory_topology::unified, no_managed),
                 std::invalid_argument);
}

TEST(HEonGPU, MemoryPool_Reads_The_Memory_Mode_From_The_Environment)
{
    using heongpu::memory_mode;
    using heongpu::detail::memory_mode_from_environment;
    EXPECT_EQ(memory_mode_from_environment(nullptr, memory_mode::copy),
              memory_mode::copy);
    EXPECT_EQ(memory_mode_from_environment("", memory_mode::unified),
              memory_mode::unified);
    EXPECT_EQ(memory_mode_from_environment("automatic", memory_mode::copy),
              memory_mode::automatic);
    EXPECT_EQ(memory_mode_from_environment("copy", memory_mode::unified),
              memory_mode::copy);
    EXPECT_EQ(memory_mode_from_environment("unified", memory_mode::copy),
              memory_mode::unified);
    EXPECT_THROW(memory_mode_from_environment("managed", memory_mode::copy),
                 std::invalid_argument);
}

TEST(HEonGPU, MemoryPool_Allocates_Managed_Memory_In_The_Unified_Mode)
{
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    const auto attributes =
        heongpu::detail::query_device_memory_attributes(device);
    if (attributes.concurrent_managed_access == 0)
    {
        GTEST_SKIP() << "The GPU has no concurrent managed access.";
    }
    auto& pool = heongpu::MemoryPool::instance();

    for (heongpu::memory_mode requested :
         {heongpu::memory_mode::unified, heongpu::memory_mode::copy})
    {
        heongpu::shutdown();
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});
        heongpu::MemoryPoolConfig pool_config;
        pool_config.initial_device_bytes = 256ULL << 20;
        pool_config.max_device_bytes = 1024ULL << 20;
        pool_config.memory_mode = requested;
        context->generate(pool_config);

        // HEONGPU_MEMORY_MODE overrides the configuration.
        const heongpu::memory_mode expected =
            heongpu::detail::select_memory_mode(
                heongpu::detail::memory_mode_from_environment(
                    std::getenv("HEONGPU_MEMORY_MODE"), requested),
                pool.topology(), attributes);
        EXPECT_EQ(pool.mode(), expected);

        heongpu::DeviceVector<double> device_vector(count);
        cudaPointerAttributes pointer;
        ASSERT_EQ(cudaPointerGetAttributes(&pointer, device_vector.data()),
                  cudaSuccess);
        EXPECT_EQ(pointer.type, expected == heongpu::memory_mode::unified
                                    ? cudaMemoryTypeManaged
                                    : cudaMemoryTypeDevice);

        std::ostringstream status;
        pool.print_memory_pool_status(status);
        EXPECT_NE(status.str().find(expected == heongpu::memory_mode::unified
                                        ? "Mode        : unified"
                                        : "Mode        : copy"),
                  std::string::npos)
            << status.str();

        // Encryption and decryption work in both modes.
        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);
        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);
        const std::vector<double> message(2048, 0.5);
        heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext(context);
        encoder.encode(plaintext, message, std::pow(2.0, 30));
        heongpu::Ciphertext<heongpu::Scheme::CKKS> ciphertext(context);
        encryptor.encrypt(ciphertext, plaintext);
        heongpu::Plaintext<heongpu::Scheme::CKKS> decrypted(context);
        decryptor.decrypt(decrypted, ciphertext);
        std::vector<double> result;
        encoder.decode(result, decrypted);
        ASSERT_EQ(result.size(), message.size());
        for (size_t i = 0; i < message.size(); i++)
        {
            EXPECT_NEAR(result[i], message[i], 1e-3);
        }
    }
    heongpu::shutdown();
}

TEST(HEonGPU, MemoryPool_Unified_Mode_Keeps_Objects_In_Device_Memory)
{
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    if (heongpu::detail::query_device_memory_attributes(device)
            .concurrent_managed_access == 0)
    {
        GTEST_SKIP() << "The GPU has no concurrent managed access.";
    }
    using Ciphertext = heongpu::Ciphertext<heongpu::Scheme::CKKS>;
    auto& pool = heongpu::MemoryPool::instance();
    heongpu::shutdown();
    {
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});
        heongpu::MemoryPoolConfig pool_config;
        pool_config.initial_device_bytes = 256ULL << 20;
        pool_config.max_device_bytes = 1024ULL << 20;
        pool_config.memory_mode = heongpu::memory_mode::unified;
        context->generate(pool_config);
        if (pool.mode() != heongpu::memory_mode::unified)
        {
            GTEST_SKIP() << "HEONGPU_MEMORY_MODE selects another mode.";
        }

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);
        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                              public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                              secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                       encoder);

        const std::vector<double> message(2048, 0.25);
        heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext(context);
        encoder.encode(plaintext, message, std::pow(2.0, 30));
        Ciphertext ciphertext(context);
        encryptor.encrypt(ciphertext, plaintext);
        std::vector<Data64> expected;
        ciphertext.get_data(expected);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // Storing in host memory neither copies nor allocates: the host
        // reads the data where it is.
        const size_t device_used = pool.get_current_device_pool_memory_usage();
        const size_t host_used = pool.get_current_host_pool_memory_usage();
        ciphertext.store_in_host();
        EXPECT_TRUE(ciphertext.is_on_device());
        EXPECT_EQ(pool.get_current_device_pool_memory_usage(), device_used);
        EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_used);
        const Data64* data = ciphertext.data();
        EXPECT_EQ(std::vector<Data64>(data, data + expected.size()), expected);

        // So does an operation that asks for host memory.
        Ciphertext sum(context);
        operators.add(ciphertext, ciphertext, sum,
                      heongpu::ExecutionOptions().set_storage_type(
                          heongpu::storage_type::host));
        EXPECT_TRUE(sum.is_on_device());
        EXPECT_EQ(pool.get_current_host_pool_memory_usage(), host_used);

        heongpu::Plaintext<heongpu::Scheme::CKKS> decrypted(context);
        decryptor.decrypt(decrypted, sum);
        std::vector<double> result;
        encoder.decode(result, decrypted);
        ASSERT_EQ(result.size(), message.size());
        for (size_t i = 0; i < message.size(); i++)
        {
            EXPECT_NEAR(result[i], 2 * message[i], 1e-3);
        }
    }
    heongpu::shutdown();
}

TEST(HEonGPU, MemoryPool_Unified_Mode_Serializes_Without_Host_Memory)
{
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    if (heongpu::detail::query_device_memory_attributes(device)
            .concurrent_managed_access == 0)
    {
        GTEST_SKIP() << "The GPU has no concurrent managed access.";
    }
    auto& pool = heongpu::MemoryPool::instance();
    heongpu::shutdown();
    {
        heongpu::HEContext<heongpu::Scheme::CKKS> context =
            heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none);
        context->set_poly_modulus_degree(4096);
        context->set_coeff_modulus_bit_sizes({40, 30}, {40});
        heongpu::MemoryPoolConfig pool_config;
        pool_config.initial_device_bytes = 256ULL << 20;
        pool_config.max_device_bytes = 1024ULL << 20;
        pool_config.memory_mode = heongpu::memory_mode::unified;
        context->generate(pool_config);
        if (pool.mode() != heongpu::memory_mode::unified)
        {
            GTEST_SKIP() << "HEONGPU_MEMORY_MODE selects another mode.";
        }

        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // The host pool serves no staging buffer while the key is saved and
        // loaded.
        auto host_allocations = [&]
        { return pool.get_host_resource()->get_bytes_counter().total; };
        const auto before = host_allocations();
        std::stringstream saved;
        relin_key.save(saved);
        heongpu::Relinkey<heongpu::Scheme::CKKS> loaded(context);
        std::stringstream input(saved.str());
        loaded.load(input);
        std::stringstream saved_again;
        loaded.save(saved_again);
        EXPECT_EQ(host_allocations(), before);
        EXPECT_EQ(saved_again.str(), saved.str());
    }
    heongpu::shutdown();
}

TEST(HEonGPU, MemoryPool_Selects_The_Device_Pool_Kind)
{
    using heongpu::memory_mode;
    using heongpu::memory_topology;
    using heongpu::detail::device_pool_kind;
    using heongpu::detail::select_device_pool_kind;
    EXPECT_EQ(select_device_pool_kind(memory_topology::discrete,
                                      memory_mode::copy, true),
              device_pool_kind::growing);
    EXPECT_EQ(select_device_pool_kind(memory_topology::unified,
                                      memory_mode::copy, true),
              device_pool_kind::releasing);
    EXPECT_EQ(select_device_pool_kind(memory_topology::discrete,
                                      memory_mode::unified, true),
              device_pool_kind::releasing);
    EXPECT_EQ(select_device_pool_kind(memory_topology::unified,
                                      memory_mode::unified, false),
              device_pool_kind::growing);
}

TEST(HEonGPU, MemoryPool_Releasing_Device_Pool_Returns_Unused_Memory)
{
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int pools_supported = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&pools_supported,
                                     cudaDevAttrMemoryPoolsSupported, device),
              cudaSuccess);
    if (pools_supported == 0)
    {
        GTEST_SKIP() << "The GPU has no stream-ordered memory pools.";
    }
    const bool managed = heongpu::detail::query_device_memory_attributes(device)
                             .concurrent_managed_access != 0;
    auto& pool = heongpu::MemoryPool::instance();
    constexpr size_t threshold = 64ULL << 20;

    for (heongpu::memory_mode mode :
         {heongpu::memory_mode::copy, heongpu::memory_mode::unified})
    {
        if (mode == heongpu::memory_mode::unified && !managed)
        {
            continue;
        }
        SCOPED_TRACE(static_cast<int>(mode));
        heongpu::shutdown();
        heongpu::detail::override_device_pool_kind(
            heongpu::detail::device_pool_kind::releasing);
        {
            heongpu::HEContext<heongpu::Scheme::CKKS> context =
                heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                    heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(4096);
            context->set_coeff_modulus_bit_sizes({40, 30}, {40});
            heongpu::MemoryPoolConfig pool_config;
            pool_config.initial_device_bytes = 64ULL << 20;
            pool_config.max_device_bytes = 2ULL << 30;
            pool_config.device_release_threshold_bytes = threshold;
            pool_config.memory_mode = mode;
            context->generate(pool_config);
            if (pool.device_pool_kind() !=
                heongpu::detail::device_pool_kind::releasing)
            {
                // A CUDA version without pools of managed memory.
                continue;
            }

            {
                heongpu::DeviceVector<Data64> large((1ULL << 30) /
                                                    sizeof(Data64));
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                EXPECT_GE(pool.get_free_device_pool_memory() +
                              pool.get_current_device_pool_memory_usage(),
                          1ULL << 30);
            }
            // The memory beyond the threshold returns to the system at the
            // synchronization.
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
#ifdef __HIP_PLATFORM_AMD__
            // HIP reserves pool memory in blocks of 128 MiB and keeps a block
            // while any allocation in it is live.
            constexpr size_t slack = 128ULL << 20;
#else
            constexpr size_t slack = 32ULL << 20;
#endif
            EXPECT_LE(pool.get_free_device_pool_memory() +
                          pool.get_current_device_pool_memory_usage(),
                      threshold + slack);

            // The pool stays within its max size.
            EXPECT_THROW(
                heongpu::DeviceVector<Data64>((3ULL << 30) / sizeof(Data64)),
                rmm::out_of_memory);

            std::ostringstream status;
            pool.print_memory_pool_status(status);
            EXPECT_NE(status.str().find("Kind      : releasing"),
                      std::string::npos)
                << status.str();
        }
        heongpu::shutdown();
    }
    heongpu::detail::override_device_pool_kind(std::nullopt);
}
