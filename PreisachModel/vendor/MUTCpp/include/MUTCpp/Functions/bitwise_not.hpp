// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Compute the bit-wise NOT the input array element-wise.
    template<typename dtype>
    Matrix<dtype> bitwise_not(const Matrix<dtype>& inArray)
    {
        return ~inArray;
    }
}
