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
    T bitwise_xor(T left, T right)
    {
        return static_cast<T>(left ^ right);
    }

    template <typename Left, typename Right>
    auto bitwise_xor(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::bitwise_xor(a, static_cast<decltype(a)>(b));
        });
    }
}
