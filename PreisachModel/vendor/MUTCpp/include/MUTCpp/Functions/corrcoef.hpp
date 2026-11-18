// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/cov.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixXd corrcoef(const Eigen::DenseBase<Derived> &input)
    {
        Eigen::MatrixXd covariance = cov(input);
        const Eigen::VectorXd deviation = covariance.diagonal().array().sqrt();
        return (covariance.array() / (deviation * deviation.transpose()).array()).matrix();
    }
}
