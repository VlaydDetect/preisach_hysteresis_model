// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

namespace mc
{
    template <typename Left, typename Right>
    Left &copyto(Eigen::DenseBase<Left> &destination, const Eigen::DenseBase<Right> &right)
    {
        destination.derived() = right.derived().eval();
        return destination.derived();
    }
}
