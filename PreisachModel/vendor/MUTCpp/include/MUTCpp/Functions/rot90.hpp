// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/flip.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> rot90(const Eigen::DenseBase<Derived> &input, unsigned turns = 1)
    {
        const auto &values = input.derived();
        switch (turns % 4)
        {
        case 0: return values;
        case 1: return flipud(values.transpose());
        case 2: return flip(values);
        default: return fliplr(values.transpose());
        }
    }
}
