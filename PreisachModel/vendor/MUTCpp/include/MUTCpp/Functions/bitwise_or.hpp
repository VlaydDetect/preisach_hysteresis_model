// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Compute the bit-wise OR of two arrays element-wise.
    template<typename dtype>
    Matrix<dtype> bitwise_or(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        return inArray1 | inArray2;
    }
}
