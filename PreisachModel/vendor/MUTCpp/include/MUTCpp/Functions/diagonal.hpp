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
    Eigen::RowVectorX<typename Derived::Scalar> diagonal(const Eigen::DenseBase<Derived> &input, Eigen::Index offset = 0, Axis axis = Axis::ROW)
    {
        const auto &value = input.derived();
        if (axis == Axis::COL) offset = -offset;
        if (offset <= -input.rows() || offset >= input.cols()) return {};
        return value.matrix().diagonal(offset).transpose();
    }
}
