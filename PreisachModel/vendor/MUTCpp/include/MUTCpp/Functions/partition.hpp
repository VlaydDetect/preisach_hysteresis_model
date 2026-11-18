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
    Eigen::MatrixX<typename Derived::Scalar> partition(const Eigen::DenseBase<Derived> &input, Eigen::Index kth, Axis axis = Axis::NONE)
    {
        auto transformed = detail::mapAxis(input, axis, 0, [&](const auto &slice)
        {
            Eigen::RowVectorX<typename Derived::Scalar> values = detail::rowMajor(slice).transpose();
            if (kth < 0 || kth >= values.size()) throw std::out_of_range("Partition index");
            std::nth_element(values.begin(), values.begin() + kth, values.end());
            return values;
        });
        if (axis == Axis::NONE)
        {
            return transformed.template reshaped<Eigen::RowMajor>(input.rows(), input.cols()).eval();
        }
        return transformed;
    }
}
