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
    Eigen::MatrixX<typename Derived::Scalar> full_like(const Eigen::DenseBase<Derived> &input, const typename Derived::Scalar &value)
    {
        return Eigen::MatrixX<typename Derived::Scalar>::Constant(input.rows(), input.cols(), value);
    }
}
