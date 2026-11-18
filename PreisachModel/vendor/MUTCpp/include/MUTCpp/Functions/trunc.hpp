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
    /// Return the truncated value of the input.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    dtype trunc(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::trunc(inValue);
    }

    /// Return the truncated value of the input, element-wise.
    template <typename Derived>
    auto trunc(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::trunc(inValue);
        }).eval();
    }
}
