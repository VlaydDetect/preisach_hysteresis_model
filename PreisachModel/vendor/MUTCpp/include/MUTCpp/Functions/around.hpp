// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include "Functions/round.hpp"

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    auto around(T value, unsigned decimals = 0)
    {
        return mc::round(value, decimals);
    }

    template <typename Derived>
    auto around(const Eigen::DenseBase<Derived> &input, unsigned decimals = 0)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::round(value, decimals);
        }).eval();
    }
}
