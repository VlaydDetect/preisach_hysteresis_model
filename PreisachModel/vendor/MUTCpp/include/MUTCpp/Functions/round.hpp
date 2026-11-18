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
    auto round(T value, unsigned decimals = 0)
    {
        const auto scale = std::pow(10.0, decimals);
        return static_cast<T>(std::round(value * scale) / scale);
    }

    template <typename Derived>
    auto round(const Eigen::DenseBase<Derived> &input, unsigned decimals = 0)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::round(value, decimals);
        }).eval();
    }
}
