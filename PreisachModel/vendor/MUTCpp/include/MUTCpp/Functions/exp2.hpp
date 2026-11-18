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
    /// Calculate 2**p for all p in the input value.
    template<typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto exp2(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::exp2(inValue);
    }
    
    /// Calculate 2**p for all p in the input array.
    template <typename Derived>
    auto exp2(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::exp2(inValue);
        }).eval();
    }
}
