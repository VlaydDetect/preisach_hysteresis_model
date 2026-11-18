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
    Derived &swapRows(Eigen::DenseBase<Derived> &input, Eigen::Index left, Eigen::Index right)
    {
        if (left < 0) left += input.rows();
        if (right < 0) right += input.rows();
        if (left < 0 || right < 0 || left >= input.rows() || right >= input.rows())
        {
            throw std::out_of_range("Swap index out of range");
        }
        input.derived().row(left).swap(input.derived().row(right));
        return input.derived();
    }
}
