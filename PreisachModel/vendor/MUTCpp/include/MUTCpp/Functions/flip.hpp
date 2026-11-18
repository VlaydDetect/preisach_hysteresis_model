// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> flip(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        const auto &values = input.derived();
        switch (axis)
        {
        case Axis::NONE: return values.reverse();
        case Axis::COL: return values.rowwise().reverse();
        case Axis::ROW: return values.colwise().reverse();
        }
        throw std::invalid_argument("Unsupported axis");
    }

    template <typename Derived>
    auto flipud(const Eigen::DenseBase<Derived> &input)
    {
        return flip(input, Axis::ROW);
    }

    template <typename Derived>
    auto fliplr(const Eigen::DenseBase<Derived> &input)
    {
        return flip(input, Axis::COL);
    }
}
