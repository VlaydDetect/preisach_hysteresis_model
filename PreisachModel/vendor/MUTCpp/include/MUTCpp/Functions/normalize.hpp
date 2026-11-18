// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/norm.hpp"

namespace mc
{
    template <typename Derived>
    auto normalize(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using Result = std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>;
        Eigen::MatrixX<Result> result = input.derived().matrix().template cast<Result>();
        const auto lengths = mc::norm(input, axis);
        switch (axis)
        {
        case Axis::NONE:
            result /= lengths(0, 0);
            break;
        case Axis::COL:
            result.array().colwise() /= lengths.row(0).transpose().array();
            break;
        case Axis::ROW:
            result.array().rowwise() /= lengths.row(0).array();
            break;
        }
        return result;
    }
}
