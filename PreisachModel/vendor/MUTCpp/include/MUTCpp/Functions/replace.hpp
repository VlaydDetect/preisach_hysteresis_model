// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    template <typename Derived>
    auto replace(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar oldValue, typename Derived::Scalar newValue)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return utils::essentiallyEqual(value, oldValue) ? newValue : value;
        }).eval();
    }
}
