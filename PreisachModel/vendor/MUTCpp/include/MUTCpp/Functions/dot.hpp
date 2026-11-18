// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/inner.hpp"

namespace mc
{
    template <typename Left, typename Right>
    Eigen::MatrixX<std::common_type_t<typename Left::Scalar, typename Right::Scalar>> dot(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        using Scalar = std::common_type_t<typename Left::Scalar, typename Right::Scalar>;
        if (left.rows() == right.rows() && left.cols() == right.cols() && (left.rows() == 1 || left.cols() == 1))
        {
            return Eigen::MatrixX<Scalar>::Constant(1, 1, inner(left, right)).eval();
        }
        if (left.cols() != right.rows()) throw std::invalid_argument("Product dimensions differ");
        const Eigen::MatrixX<Scalar> a = left.derived().matrix().template cast<Scalar>();
        const Eigen::MatrixX<Scalar> b = right.derived().matrix().template cast<Scalar>();
        return a * b;
    }
}
