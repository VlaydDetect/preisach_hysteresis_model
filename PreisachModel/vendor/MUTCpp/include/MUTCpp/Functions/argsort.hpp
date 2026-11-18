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
    Eigen::MatrixX<Eigen::Index> argsort(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::mapAxis(input, axis, 0, [](const auto &values)
        {
            Eigen::RowVectorX<Eigen::Index> indices(values.size());
            std::iota(indices.begin(), indices.end(), Eigen::Index{});
            std::stable_sort(indices.begin(), indices.end(), [&](Eigen::Index a, Eigen::Index b)
            {
                return values(a) < values(b);
            });
            return indices;
        });
    }
}
