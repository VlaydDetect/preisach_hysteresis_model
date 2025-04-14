// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>

#include "Core/Error.hpp"

namespace mc
{
    inline std::vector<int32> range(int32 start, int32 stop, int32 step)
    {
        if (step == 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("Step for range must be non-zero");
        }

        std::vector<int32> result;
        int32 i = start;
        while (step > 0 ? i < stop : i > stop)
        {
            result.push_back(i);
            i += step;
        }

        return result;
    }

    inline std::vector<int32> range(int32 start, int32 stop)
    {
        return range(start, stop, 1);
    }

    inline std::vector<int32> range(int32 stop)
    {
        return range(0, stop, 1);
    }
}
