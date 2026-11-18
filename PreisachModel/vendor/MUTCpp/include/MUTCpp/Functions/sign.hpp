// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/Types.hpp"
#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Returns an element-wise indication of the sign of a number.
    /// The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    /// nan is returned for nan inputs.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    int8 sign(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inValue < dtype{0})
        {
            return -1;
        }

        if (inValue > dtype{0})
        {
            return 1;
        }

        return 0;
    }

    /// Returns an element-wise indication of the sign of a number.
    /// The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    /// nan is returned for nan inputs.
    template <typename Derived>
    auto sign(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::sign(inValue);
        }).eval();
    }
}
