// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include <Eigen/Core>

#include "Core/StaticAssert.hpp"

namespace mc
{
    /// Natural logarithm.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    auto log(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(inValue);
    }

    /// Natural logarithm.
    template <typename dtype>
    auto log(std::complex<dtype> inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(inValue);
    }

    /// Natural logarithm, element-wise.
    template <typename Derived>
    auto log(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::log(inValue);
        }).eval();
    }
}
