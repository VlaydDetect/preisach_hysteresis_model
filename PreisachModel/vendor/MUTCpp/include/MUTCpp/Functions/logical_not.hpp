// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    /// Compute the truth value of NOT x element-wise.
    template <typename Derived>
    auto logical_not(const Eigen::DenseBase<Derived> &inArray)
    {
        using Scalar = typename Derived::Scalar;
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return utils::essentiallyEqual(inValue, Scalar{0});
        }).eval();
    }
}
