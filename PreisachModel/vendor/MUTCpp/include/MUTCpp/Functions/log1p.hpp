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
    /// Return the natural logarithm of one plus the input array.
    /// Calculates log(1 + x).
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto log1p(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::log1p(inValue);
    }

    /// Return the natural logarithm of one plus the input array, element-wise.
    /// Calculates log(1 + x).
    template <typename Derived>
    auto log1p(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::log1p(inValue);
        }).eval();
    }
}
