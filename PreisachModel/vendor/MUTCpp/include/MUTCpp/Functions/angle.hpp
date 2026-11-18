// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <complex>

namespace mc
{
    template <typename T>
    auto angle(const std::complex<T> &value)
    {
        return std::arg(value);
    }

    template <typename Derived>
    auto angle(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return std::arg(value);
        }).eval();
    }
}
