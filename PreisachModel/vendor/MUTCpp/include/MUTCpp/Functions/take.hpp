// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    template <typename Derived, typename Indices>
    requires std::is_integral_v<typename Indices::Scalar>
    Eigen::MatrixX<typename Derived::Scalar> take(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices, Axis axis = Axis::NONE)
    {
        const auto size = detail::axisSize(input.rows(), input.cols(), axis);
        const Eigen::VectorX<Eigen::Index> positions = detail::rowMajor(indices).unaryExpr([size](auto index)
        {
            return detail::checkedIndex(static_cast<Eigen::Index>(index), size);
        });
        const auto &values = input.derived();
        if (axis == Axis::ROW) return values(positions, Eigen::all).matrix();
        if (axis == Axis::COL) return values(Eigen::all, positions).matrix();
        return detail::rowMajor(input)(positions).transpose().matrix();
    }
}
