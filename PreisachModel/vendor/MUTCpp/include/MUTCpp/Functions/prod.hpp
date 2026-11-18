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
    auto prod(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using Scalar = typename Derived::Scalar;
        return detail::reduce(input, axis, [](const auto &values)
        {
            return std::accumulate(values.begin(), values.end(), Scalar{1}, std::multiplies<>{});
        });
    }
}
