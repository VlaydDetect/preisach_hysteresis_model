// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.
#pragma once

#include <limits>

namespace mc
{
    namespace hash_detail
    {
        template <class T>
        struct limits : std::numeric_limits<T>
        {
        };
    }
}
