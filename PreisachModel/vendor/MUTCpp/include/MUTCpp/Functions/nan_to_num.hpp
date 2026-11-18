// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> nan_to_num(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar nan = {},
        typename Derived::Scalar posInf = (std::numeric_limits<typename Derived::Scalar>::max)(),
        typename Derived::Scalar negInf = (std::numeric_limits<typename Derived::Scalar>::min)())
    {
        return input.derived().matrix().unaryExpr([=](const auto value)
        {
            if (std::isnan(value)) return nan;
            if (std::isinf(value)) return value > 0 ? posInf : negInf;
            return value;
        });
    }
}
