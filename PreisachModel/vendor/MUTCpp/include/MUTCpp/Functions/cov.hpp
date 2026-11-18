// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Core/Enums.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixXd cov(const Eigen::DenseBase<Derived> &input, Bias bias = Bias::NO)
    {
        const Eigen::MatrixXd data = input.derived().template cast<double>().matrix();
        const Eigen::VectorXd means = data.rowwise().mean();
        const Eigen::MatrixXd centered = data.colwise() - means;
        const auto divisor = data.cols() - (bias == Bias::NO);
        return centered * centered.transpose() / static_cast<double>(divisor);
    }
}
