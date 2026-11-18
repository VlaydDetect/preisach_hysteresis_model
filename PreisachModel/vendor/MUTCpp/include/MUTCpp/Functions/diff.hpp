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
    auto diff(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::mapAxis(input, axis, -1, [&](const auto &slice)
        {
            using T = typename Derived::Scalar;
            Eigen::RowVectorX<T> result((std::max)(Eigen::Index{}, slice.size() - 1));
            for (Eigen::Index i = 0; i < result.size(); ++i) result(i) = slice(i + 1) - slice(i);
            return result;
        });
    }
}
