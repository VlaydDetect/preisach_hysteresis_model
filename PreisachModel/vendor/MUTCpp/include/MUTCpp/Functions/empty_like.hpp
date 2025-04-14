// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return a new array with the same shape as a given array.
    template<typename dtype>
    Matrix<dtype> empty_like(const Matrix<dtype>& inArray)
    {
        return Matrix<dtype>(inArray.shape());
    }
}
