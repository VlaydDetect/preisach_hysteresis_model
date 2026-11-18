// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/isclose.hpp"

namespace mc
{
    template <typename Left, typename Right>
    bool allclose(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, double rtol = 1e-5, double atol = 1e-8)
    {
        return isclose(left, right, rtol, atol).all();
    }
}
