// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <limits>

namespace mc
{
    template <typename Derived>
    Eigen::MatrixXd nans_like(const Eigen::DenseBase<Derived> &input)
    {
        return Eigen::MatrixXd::Constant(input.rows(), input.cols(), std::numeric_limits<double>::quiet_NaN());
    }
}
