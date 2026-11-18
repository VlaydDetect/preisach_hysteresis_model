// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "MUTCpp/Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Trigonometric inverse sine.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto arcsin(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::asin(inValue);
    }

    /// Trigonometric inverse sine, element-wise.
    template <typename Derived>
    auto arcsin(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::arcsin(inValue);
        }).eval();
    }
}
