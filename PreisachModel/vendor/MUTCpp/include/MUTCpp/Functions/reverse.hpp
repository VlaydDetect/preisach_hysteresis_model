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
    auto reverse(const Eigen::DenseBase<Derived> &input)
    {
        if (input.rows() != 1 && input.cols() != 1)
        {
            throw std::invalid_argument("Reverse requires a row or column vector");
        }
        return input.derived().reverse().eval();
    }
}
