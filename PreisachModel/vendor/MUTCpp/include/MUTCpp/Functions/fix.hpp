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
    /// Round to nearest integer towards zero.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    dtype fix(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue > 0 ? std::floor(inValue) : std::ceil(inValue);
    }

    /// Round to nearest integer towards zero.
\
    template <typename Derived>
    auto fix(const Eigen::DenseBase<Derived> &inArray)
    {
        using Scalar = typename Derived::Scalar;
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return static_cast<Scalar>(mc::fix(inValue));
        }).eval();
    }
}
