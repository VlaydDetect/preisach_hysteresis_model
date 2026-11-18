// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cmath>

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    T floor_divide(T left, T right)
    {
        return static_cast<T>(std::floor(static_cast<double>(left) / right));
    }

    template <typename Left, typename Right>
    auto floor_divide(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::floor_divide(a, static_cast<decltype(a)>(b));
        });
    }
}
