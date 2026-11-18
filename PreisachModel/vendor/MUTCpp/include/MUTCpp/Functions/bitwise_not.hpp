// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

namespace mc
{
    template <typename Derived>
    auto bitwise_not(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return static_cast<typename Derived::Scalar>(~value);
        }).eval();
    }
}
