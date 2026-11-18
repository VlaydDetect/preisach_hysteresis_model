// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <utility>

namespace mc
{
    template <typename Derived, typename Function>
    void applyFunction(Eigen::DenseBase<Derived> &input, Function function)
    {
        input.derived() = input.derived().unaryExpr(std::move(function)).eval();
    }
}
