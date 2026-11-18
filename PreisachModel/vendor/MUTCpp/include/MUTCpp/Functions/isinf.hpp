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
    /// Test for inf and return result as a boolean.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    bool isinf(dtype inValue) noexcept
    {
        STATIC_ASSERT_FLOAT(dtype);

        return std::isinf(inValue);
    }

    /// Test element-wise for inf and return result as a boolean array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isinf.html
    ///
    /// @param inArray
    ///
    /// @return Matrix
    ///
    template <typename Derived>
    auto isinf(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::isinf(inValue);
        }).eval();
    }
}
