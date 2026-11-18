// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/diagonal.hpp"

namespace mc
{
    template <typename Derived>
    typename Derived::Scalar trace(const Eigen::DenseBase<Derived> &input, Eigen::Index offset = 0, Axis axis = Axis::ROW)
    {
        return diagonal(input, offset, axis).sum();
    }
}
