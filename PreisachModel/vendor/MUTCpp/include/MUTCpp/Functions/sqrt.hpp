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
    /// Return the positive square-root of a value.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto sqrt(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::sqrt(inValue);
    }

    /// Return the positive square-root of an array, element-wise.
    template <typename Derived>
    auto sqrt(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::sqrt(inValue);
        }).eval();
    }
}
