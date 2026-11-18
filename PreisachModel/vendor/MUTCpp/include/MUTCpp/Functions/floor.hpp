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
    /// Return the floor of the input.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    dtype floor(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::floor(inValue);
    }

    /// Return the floor of the input, element-wise.
    template <typename Derived>
    auto floor(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::floor(inValue);
        }).eval();
    }
}
