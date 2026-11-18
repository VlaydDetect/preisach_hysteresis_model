// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/mean.hpp"

namespace mc
{
    template <typename Derived>
    auto average(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return mc::mean(input, axis);
    }

    template <typename Derived, typename Weights>
    auto average(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Weights> &weights, Axis axis = Axis::NONE)
    {
        using Result = std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>;
        if ((axis == Axis::NONE && (input.rows() != weights.rows() || input.cols() != weights.cols())) ||
            (axis == Axis::COL && input.cols() != weights.size()) ||
            (axis == Axis::ROW && input.rows() != weights.size()))
        {
            throw std::invalid_argument("Array and weights are incompatible");
        }
        const auto weight = detail::rowMajor(weights).template cast<double>().eval();
        const double total = weight.sum();
        return detail::reduce(input, axis, [&](const auto &values)
        {
            Result sum{};
            for (Eigen::Index i = 0; i < values.size(); ++i)
            {
                sum += static_cast<Result>(values(i)) * weight(i);
            }
            return sum / total;
        });
    }
}
