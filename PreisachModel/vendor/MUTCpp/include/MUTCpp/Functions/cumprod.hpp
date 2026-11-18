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
    auto cumprod(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::cumulative(input, axis, std::multiplies<>{});
    }
}
