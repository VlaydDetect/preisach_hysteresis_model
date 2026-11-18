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
    double hypot(T left, T right)
    {
        return std::hypot(left, right);
    }

    template <typename Left, typename Right>
    auto hypot(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::hypot(a, static_cast<decltype(a)>(b));
        });
    }

    template <typename T>
    double hypot(T a, T b, T c)
    {
        return std::hypot(a, b, c);
    }

    template <typename Left, typename Right, typename Third>
    auto hypot(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, const Eigen::DenseBase<Third> &third)
    {
        const auto planar = hypot(left, right);
        return detail::broadcast(planar, third, [](double a, auto b)
        {
            return std::hypot(a, b);
        });
    }
}
