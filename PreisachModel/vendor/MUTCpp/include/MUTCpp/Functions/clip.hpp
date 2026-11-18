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
    auto clip(T value, T low, T high)
    {
        return std::clamp(value, low, high);
    }

    template <typename Derived>
    auto clip(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar low, typename Derived::Scalar high)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return std::clamp(value, low, high);
        }).eval();
    }
}
