// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/array_equiv.hpp"

namespace mc
{
    template <typename Left, typename Right>
    bool array_equal(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return left.rows() == right.rows() && left.cols() == right.cols() && array_equiv(left, right);
    }
}
