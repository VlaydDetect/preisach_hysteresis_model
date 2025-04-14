// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.
#pragma once

#include <cstdint>
# include <intrin.h>

namespace mc
{
    namespace hash_detail
    {
        __forceinline std::uint64_t mulx(std::uint64_t x, std::uint64_t y)
        {
            std::uint64_t r2;
            std::uint64_t r = _umul128(x, y, &r2);
            return r ^ r2;
        }
    }
}
