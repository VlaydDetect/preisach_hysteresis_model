// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cmath>

namespace mc
{
    template <typename Left, typename Right>
    auto isclose(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, double rtol = 1e-5, double atol = 1e-8)
    {
        if (left.rows() != right.rows() || left.cols() != right.cols())
        {
            throw std::invalid_argument("Array shapes must match");
        }
        return detail::broadcast(left, right, [=](auto a, auto b)
        {
            return std::abs(a - b) <= atol + rtol * std::abs(b);
        });
    }
}
