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
    auto nth_root(T value, double root)
    {
        return std::pow(value, 1.0 / root);
    }

    template <typename Derived>
    auto nth_root(const Eigen::DenseBase<Derived> &input, double root)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::nth_root(value, root);
        }).eval();
    }
}
