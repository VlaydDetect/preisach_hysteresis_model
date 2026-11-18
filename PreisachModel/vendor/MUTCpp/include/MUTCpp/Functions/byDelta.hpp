// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <Eigen/Core>
#include "Core/Types.hpp"

namespace mc
{
    template<typename dtype>
    Eigen::RowVectorX<dtype> byDelta(dtype first, dtype last, dtype delta)
    {
        auto steps = static_cast<uint32>((last - first) / delta);
        Eigen::RowVectorX<dtype> res(steps);
        for (auto i = 0; i < steps; ++i)
        {
            res[i] = first + i * delta;
        }
        return res;
    }
}
