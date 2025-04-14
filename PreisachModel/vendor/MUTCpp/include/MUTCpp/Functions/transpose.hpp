// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Permute the dimensions of an array.
    template<typename dtype>
    Matrix<dtype> transpose(const Matrix<dtype>& inArray)
    {
        return inArray.transpose();
    }
}
