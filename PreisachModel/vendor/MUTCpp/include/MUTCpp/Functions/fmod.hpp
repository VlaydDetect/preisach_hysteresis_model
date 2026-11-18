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
    T fmod(T left, T right)
    {
        if constexpr (std::is_integral_v<T>) return left % right;
        else return static_cast<T>(std::fmod(left, right));
    }

    template <typename Left, typename Right>
    auto fmod(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::fmod(a, static_cast<decltype(a)>(b));
        });
    }
}
