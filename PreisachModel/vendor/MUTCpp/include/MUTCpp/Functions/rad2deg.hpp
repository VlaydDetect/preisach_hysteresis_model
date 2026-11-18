// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Constants.hpp"
#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Convert angles from radians to degrees.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    constexpr auto rad2deg(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue * 180. / consts::pi;
    }

    /// Convert angles from radians to degrees.
    template <typename Derived>
    auto rad2deg(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::rad2deg(inValue);
        }).eval();
    }
}
