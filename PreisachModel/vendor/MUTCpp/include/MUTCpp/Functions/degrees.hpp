// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include "Functions/rad2deg.hpp"

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    auto degrees(T value)
    {
        return mc::rad2deg(value);
    }

    template <typename Derived>
    auto degrees(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::rad2deg(value);
        }).eval();
    }
}
