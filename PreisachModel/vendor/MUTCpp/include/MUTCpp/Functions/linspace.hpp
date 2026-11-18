// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Core/Enums.hpp"
#include "Core/Types.hpp"
#include <stdexcept>

namespace mc
{
    template <typename T>
    Eigen::RowVectorX<T> linspace(T first, T last, uint32 count = 50, EndPoint endpoint = EndPoint::YES)
    {
        Eigen::RowVectorX<T> result(count);
        if (count == 0) return result;
        result(0) = first;
        if (count == 1) return result;
        if (last <= first) throw std::invalid_argument("Range end must be greater than start");
        const auto divisions = endpoint == EndPoint::YES ? count - 1 : count;
        const T step = (last - first) / static_cast<T>(divisions);
        for (uint32 i = 1; i < count; ++i)
        {
            result(i) = first + static_cast<T>(i) * step;
        }
        if (endpoint == EndPoint::YES) result(count - 1) = last;
        return result;
    }
}
