// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include <Eigen/Core>

#include "Core/StaticAssert.hpp"

namespace mc
{
    /// Trigonometric sine.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto sin(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::sin(inValue);
    }

    /// Trigonometric sine, element-wise.
    template <typename Derived>
    auto sin(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::sin(inValue);
        }).eval();
    }
}
