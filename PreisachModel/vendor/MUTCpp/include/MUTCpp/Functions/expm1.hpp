// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Calculate exp(x) - 1 for the input value.
    template<typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto expm1(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::exp(inValue) - dtype{ 1 };
    }
    
    /// Calculate exp(x) - 1 for all elements in the array.
    template <typename Derived>
    auto expm1(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::expm1(inValue);
        }).eval();
    }
}
