// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/dot.hpp"

namespace mc
{
    template <typename Left, typename Right>
    auto matmul(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return mc::dot(left, right);
    }
}
