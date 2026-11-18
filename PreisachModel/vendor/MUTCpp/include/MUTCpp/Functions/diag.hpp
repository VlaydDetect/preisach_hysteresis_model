// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/diagflat.hpp"
#include "Functions/diagonal.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> diag(const Eigen::DenseBase<Derived> &input, Eigen::Index offset = 0)
    {
        if (input.rows() == 1 || input.cols() == 1) return diagflat(input, offset);
        return diagonal(input, offset);
    }
}
