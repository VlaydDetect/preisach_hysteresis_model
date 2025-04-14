// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Swaps cols of the array
    template <typename dtype>
    Matrix<dtype> &swapCols(Matrix<dtype> &inArray, int32 colIdx1, int32 colIdx2) noexcept
    {
        inArray.swapCols(colIdx1, colIdx2);
        return inArray;
    }
}
