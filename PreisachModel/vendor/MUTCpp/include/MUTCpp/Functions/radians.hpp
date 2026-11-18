// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include "Functions/deg2rad.hpp"

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    auto radians(T value)
    {
        return mc::deg2rad(value);
    }

    template <typename Derived>
    auto radians(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::deg2rad(value);
        }).eval();
    }
}
