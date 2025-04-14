// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/full.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a full array with the same shape and type as a given array.
    template<typename dtype>
    Matrix<dtype> full_like(const Matrix<dtype>& inArray, dtype inFillValue)
    {
        return full(inArray.shape(), inFillValue);
    }
}
