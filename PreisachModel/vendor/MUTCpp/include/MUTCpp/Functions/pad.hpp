// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> pad(const Eigen::DenseBase<Derived> &input, Eigen::Index width, typename Derived::Scalar value)
    {
        if (width < 0) throw std::invalid_argument("Padding width cannot be negative");
        using T = typename Derived::Scalar;
        Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Constant(input.rows() + 2 * width, input.cols() + 2 * width, value);
        result.block(width, width, input.rows(), input.cols()) = input.derived().matrix();
        return result;
    }
}
