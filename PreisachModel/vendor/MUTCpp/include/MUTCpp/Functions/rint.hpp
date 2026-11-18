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
    /// Round value to the nearest integer.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    dtype rint(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::rint(inValue);
    }

    /// Round elements of the array to the nearest integer.
    template <typename Derived>
    auto rint(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::rint(inValue);
        }).eval();
    }
}
