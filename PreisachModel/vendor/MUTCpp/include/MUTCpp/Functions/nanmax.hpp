// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include "Functions/max.hpp"

namespace mc
{
    template <typename Derived>
    auto nanmax(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using T = typename Derived::Scalar;
        const auto cleaned = input.derived().unaryExpr([](T value)
        {
            return std::isnan(value) ? std::numeric_limits<T>::min() : value;
        }).eval();
        return mc::max(cleaned, axis);
    }
}
