// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/nanstdev.hpp"

namespace mc
{
    template <typename Derived>
    auto nanvar(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return mc::nanstdev(input, axis).array().square().matrix().eval();
    }
}
