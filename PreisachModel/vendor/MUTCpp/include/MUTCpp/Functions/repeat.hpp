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
    /// Repeat elements of an array.
    template <typename dtype>
    Matrix<dtype> repeat(const Matrix<dtype> &inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return inArray.repeat(inNumRows, inNumCols);
    }

    /// Repeat elements of an array.
    template <typename dtype>
    Matrix<dtype> repeat(const Matrix<dtype> &inArray, const Shape &inRepeatShape)
    {
        return inArray.repeat(inRepeatShape);
    }
}
