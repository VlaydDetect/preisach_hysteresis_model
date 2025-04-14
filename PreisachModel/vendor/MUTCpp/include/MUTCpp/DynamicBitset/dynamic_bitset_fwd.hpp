// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>

namespace mc
{
    template <typename Block = unsigned long,
              typename Allocator = std::allocator<Block>>
    class dynamic_bitset;
}
