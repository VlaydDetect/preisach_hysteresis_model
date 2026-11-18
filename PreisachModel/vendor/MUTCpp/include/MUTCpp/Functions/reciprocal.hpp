// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <complex>
#include "Core/TypeTraits.hpp"

namespace mc
{
    template <typename Derived>
    auto reciprocal(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return (std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>{1} / static_cast<std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>>(value));
        }).eval();
    }
}
