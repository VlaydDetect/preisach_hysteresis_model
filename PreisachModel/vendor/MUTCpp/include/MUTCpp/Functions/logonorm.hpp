// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/norm.hpp"
#include "Functions/log.hpp"

namespace mc
{
    template <typename Derived>
    auto logonorm(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return mc::log(mc::norm(input, axis));
    }
}
