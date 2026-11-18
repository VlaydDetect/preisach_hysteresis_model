// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/reshape.hpp"

namespace mc
{
    template <typename T>
    Eigen::MatrixX<T> &ravel(Eigen::MatrixX<T> &input)
    {
        return reshape(input, input.size());
    }
}
