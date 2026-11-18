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
    /// Convert angles from degrees to radians.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    constexpr auto deg2rad(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue * consts::pi / 180.;
    }

    /// Convert angles from degrees to radians.
    template <typename Derived>
    auto deg2rad(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::deg2rad(inValue);
        }).eval();
    }
}
