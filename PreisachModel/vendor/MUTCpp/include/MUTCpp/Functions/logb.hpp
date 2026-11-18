// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <algorithm>

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    auto logb(T value, T base)
    {
        return std::log(value) / std::log(base);
    }

    template <typename Derived>
    auto logb(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar base)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::logb(value, base);
        }).eval();
    }
}
