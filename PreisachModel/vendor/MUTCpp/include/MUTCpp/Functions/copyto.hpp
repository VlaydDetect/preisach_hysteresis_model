// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Copies values from one array to another
    template <typename dtype>
    Matrix<dtype> &copyto(Matrix<dtype> &inDestArray, const Matrix<dtype> &inSrcArray)
    {
        inDestArray = inSrcArray;
        return inDestArray;
    }
}
