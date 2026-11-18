// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include "Core/StaticAssert.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Return the square of an array.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    constexpr dtype square(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return utils::sqr(inValue);
    }

    /// Return the square of an array, element-wise.
    template <typename Derived>
    auto square(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::square(inValue);
        }).eval();
    }
}
