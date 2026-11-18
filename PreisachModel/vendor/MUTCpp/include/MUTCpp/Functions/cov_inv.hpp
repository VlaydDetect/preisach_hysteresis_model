// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/cov.hpp"
#include <Eigen/LU>

namespace mc
{
    template <typename Derived>
    Eigen::MatrixXd cov_inv(const Eigen::DenseBase<Derived> &input, Bias bias = Bias::NO)
    {
        return cov(input, bias).inverse();
    }
}
