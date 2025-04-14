// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Fill the main diagonal of the given array.
    template <typename dtype>
    void fillDiagonal(Matrix<dtype> &inArray, dtype inValue) noexcept
    {
        const auto inShape = inArray.shape();
        for (uint32 row = 0; row < inShape.rows; ++row)
        {
            if (row < inShape.cols)
            {
                inArray(row, row) = inValue;
            }
        }
    }
}
