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
    Derived &swapCols(Eigen::DenseBase<Derived> &input, Eigen::Index left, Eigen::Index right)
    {
        if (left < 0) left += input.cols();
        if (right < 0) right += input.cols();
        if (left < 0 || right < 0 || left >= input.cols() || right >= input.cols())
        {
            throw std::out_of_range("Swap index out of range");
        }
        input.derived().col(left).swap(input.derived().col(right));
        return input.derived();
    }
}
