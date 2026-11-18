// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Core/Enums.hpp"
#include <stdexcept>
#include <vector>

namespace mc
{
    template <typename Derived>
    [[nodiscard]] std::vector<std::vector<typename Derived::Scalar>> toVector(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::ROW)
    {
        if (input.size() == 0) return {};
        if (axis != Axis::ROW && axis != Axis::COL) throw std::invalid_argument("Unsupported axis");
        std::vector<std::vector<typename Derived::Scalar>> result;
        const auto &values = input.derived();
        if (axis == Axis::ROW)
        {
            result.reserve(values.rows());
            for (Eigen::Index i = 0; i < values.rows(); ++i)
            {
                const auto row = values.row(i);
                result.emplace_back(row.begin(), row.end());
            }
        }
        else
        {
            result.reserve(values.cols());
            for (Eigen::Index i = 0; i < values.cols(); ++i)
            {
                const auto col = values.col(i);
                result.emplace_back(col.begin(), col.end());
            }
        }
        return result;
    }
}
