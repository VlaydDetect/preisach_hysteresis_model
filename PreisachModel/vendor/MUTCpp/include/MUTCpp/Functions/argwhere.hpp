// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::RowVectorX<Eigen::Index> argwhere(const Eigen::DenseBase<Derived> &input)
    {
        using T = typename Derived::Scalar;
        const auto values = detail::rowMajor(input);
        Eigen::RowVectorX<Eigen::Index> indices(values.size());
        Eigen::Index count = 0;
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            if (!utils::essentiallyEqual(values(i), T{})) indices(count++) = i;
        }
        return indices.head(count);
    }
}
