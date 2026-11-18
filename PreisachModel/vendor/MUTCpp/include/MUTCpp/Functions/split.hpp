// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/unique.hpp"
#include <vector>

namespace mc
{
    template <typename Derived, typename Indices>
    requires std::is_integral_v<typename Indices::Scalar>
    std::vector<Eigen::MatrixX<typename Derived::Scalar>> split(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices, Axis axis = Axis::ROW)
    {
        if (axis != Axis::ROW && axis != Axis::COL)
        {
            throw std::invalid_argument("Split axis must be ROW or COL");
        }
        const auto size = detail::axisSize(input.rows(), input.cols(), axis);
        std::vector<Eigen::MatrixX<typename Derived::Scalar>> result;
        if (size == 0)
        {
            result.emplace_back(input.derived().matrix());
            return result;
        }
        const auto stops = unique(detail::rowMajor(indices).unaryExpr([size](auto index)
        {
            auto position = static_cast<Eigen::Index>(index);
            if (position < 0) position += size;
            return std::clamp(position, Eigen::Index{}, size - 1);
        }));
        Eigen::Index start = 0;
        const auto addSplit = [&](Eigen::Index stop)
        {
            if (axis == Axis::ROW) result.emplace_back(input.derived().middleRows(start, stop - start).matrix());
            else result.emplace_back(input.derived().middleCols(start, stop - start).matrix());
            start = stop;
        };
        for (const auto stop : stops) addSplit(stop);
        addSplit(size);
        return result;
    }

    template <typename Derived, typename Indices>
    requires std::is_integral_v<typename Indices::Scalar>
    auto vsplit(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices)
    {
        return split(input, indices, Axis::ROW);
    }

    template <typename Derived, typename Indices>
    requires std::is_integral_v<typename Indices::Scalar>
    auto hsplit(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices)
    {
        return split(input, indices, Axis::COL);
    }
}
