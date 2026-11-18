// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Returns element-wise True where signbit is set (less than zero).
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    bool signbit(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue < dtype{0};
    }

    /// Returns element-wise True where signbit is set (less than zero).
    template <typename Derived>
    auto signbit(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::signbit(inValue);
        }).eval();
    }
}
