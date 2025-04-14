// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Swaps rows of the array
    template<typename dtype>
    Matrix<dtype>& swapRows(Matrix<dtype>& inArray, int32 rowIdx1, int32 rowIdx2) noexcept
    {
        inArray.swapRows(rowIdx1, rowIdx2);
        return inArray;
    }
}
