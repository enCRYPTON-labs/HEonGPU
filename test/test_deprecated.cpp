// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

// The deprecated names still compile and give the results of the names that
// replace them, until 2.0 removes them.

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

TEST(Deprecated, Upper_Case_Enumerators)
{
    using heongpu::arithmetic_bootstrapping_type;
    using heongpu::encoding;
    using heongpu::logic_bootstrapping_type;
    using heongpu::storage_type;
    EXPECT_EQ(storage_type::HOST, storage_type::host);
    EXPECT_EQ(storage_type::DEVICE, storage_type::device);
    EXPECT_EQ(encoding::SLOT, encoding::slot);
    EXPECT_EQ(encoding::COEFFICIENT, encoding::coefficient);
    EXPECT_EQ(arithmetic_bootstrapping_type::NONE,
              arithmetic_bootstrapping_type::none);
    EXPECT_EQ(arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING,
              arithmetic_bootstrapping_type::regular);
    EXPECT_EQ(arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING,
              arithmetic_bootstrapping_type::slim);
    EXPECT_EQ(logic_bootstrapping_type::NONE, logic_bootstrapping_type::none);
    EXPECT_EQ(logic_bootstrapping_type::BIT_BOOTSTRAPPING,
              logic_bootstrapping_type::bit);
    EXPECT_EQ(logic_bootstrapping_type::GATE_BOOTSTRAPPING,
              logic_bootstrapping_type::gate);
}
