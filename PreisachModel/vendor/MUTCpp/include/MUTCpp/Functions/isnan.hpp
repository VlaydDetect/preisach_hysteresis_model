// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/DtypeInfo.hpp"
#include <Eigen/Core>

#include "Core/StaticAssert.hpp"

namespace mc
{
    /// Test for NaN and return result as a boolean.
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
    bool isnan(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (DtypeInfo<dtype>::isInteger())
        {
            return false;
        }

        return std::isnan(static_cast<double>(inValue)); // static_cast is needed for compiler error
    }

    //============================================================================
    // Method Description:
    /// Test element-wise for NaN and return result as a boolean array.
    template <typename Derived>
    auto isnan(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::isnan(inValue);
        }).eval();
    }
}
