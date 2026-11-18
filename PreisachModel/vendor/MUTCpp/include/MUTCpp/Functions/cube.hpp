// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include <Eigen/Core>
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Cubes the input
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    constexpr dtype cube(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return utils::cube(inValue);
    }

    /// Cubes the elements of the array
    template <typename Derived>
    auto cube(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::cube(inValue);
        }).eval();
    }
}
