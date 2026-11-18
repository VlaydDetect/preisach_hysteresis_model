// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    namespace detail
    {
        template <typename Derived, typename Spacing>
        Eigen::VectorXd trapezoidalSum(const Eigen::DenseBase<Derived> &input, Axis axis, Spacing spacing)
        {
            if (axis != Axis::NONE && axis != Axis::ROW && axis != Axis::COL)
            {
                throw std::invalid_argument("Unsupported axis");
            }
            const auto count = axis == Axis::NONE ? 1 : (axis == Axis::COL ? input.rows() : input.cols());
            const auto length = axisSize(input.rows(), input.cols(), axis);
            Eigen::VectorXd result = Eigen::VectorXd::Zero(count);
            const auto flat = rowMajor(input);
            for (Eigen::Index line = 0; line < count; ++line)
            {
                const auto sample = [&](Eigen::Index i)
                {
                    if (axis == Axis::NONE) return flat(i);
                    if (axis == Axis::COL) return input(line, i);
                    return input(i, line);
                };
                for (Eigen::Index i = 1; i < length; ++i)
                {
                    const double left = static_cast<double>(sample(i - 1));
                    const double right = static_cast<double>(sample(i));
                    result(line) += spacing(line, i) * (left + (right - left) * 0.5);
                }
            }
            return result;
        }
    }

    template <typename Derived>
    Eigen::VectorXd trapz(const Eigen::DenseBase<Derived> &input, double dx = 1.0, Axis axis = Axis::NONE)
    {
        return dx * detail::trapezoidalSum(input, axis, [](Eigen::Index, Eigen::Index) { return 1.0; });
    }

    template <typename Derived, typename Coordinates>
    Eigen::VectorXd trapz(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Coordinates> &coordinates, Axis axis = Axis::NONE)
    {
        if (input.rows() != coordinates.rows() || input.cols() != coordinates.cols())
        {
            throw std::invalid_argument("Coordinates must have the same shape as the samples");
        }
        const auto flat = detail::rowMajor(coordinates);
        const auto spacing = [&](Eigen::Index line, Eigen::Index i)
        {
            if (axis == Axis::NONE) return static_cast<double>(flat(i)) - static_cast<double>(flat(i - 1));
            if (axis == Axis::COL) return static_cast<double>(coordinates(line, i)) - static_cast<double>(coordinates(line, i - 1));
            return static_cast<double>(coordinates(i, line)) - static_cast<double>(coordinates(i - 1, line));
        };
        return detail::trapezoidalSum(input, axis, spacing);
    }
}
