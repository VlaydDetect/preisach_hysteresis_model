// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Unwrap by changing deltas between values to 2*pi complement.
    /// Unwraps to [-pi, pi].
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    dtype unwrap(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return static_cast<dtype>(std::atan2(std::sin(inValue), std::cos(inValue)));
    }

    /// Unwrap by changing deltas between values to 2*pi complement.
    /// Unwraps to [-pi, pi].
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    template <typename Derived>
    auto unwrap(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::unwrap(inValue);
        }).eval();
    }
}
