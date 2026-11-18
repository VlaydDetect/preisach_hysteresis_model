// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Functions/isinf.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Test for negative inf and return result as a boolean.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    bool isposinf(dtype inValue) noexcept
    {
        STATIC_ASSERT_FLOAT(dtype);

        return inValue > 0 && std::isinf(inValue);
    }

    /// Test element-wise for negative inf and return result as a boolean array.
    template <typename Derived>
    auto isposinf(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::isposinf(inValue);
        }).eval();
    }
}
