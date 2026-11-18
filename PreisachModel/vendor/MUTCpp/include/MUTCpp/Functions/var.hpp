// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/stdev.hpp"

namespace mc
{
    template <typename Derived>
    auto var(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return mc::stdev(input, axis).array().square().matrix().eval();
    }
}
