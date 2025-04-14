// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Construct an array by repeating the number of times given by reps.

    template <typename dtype>
    Matrix<dtype> tile(const Matrix<dtype> &inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return inArray.repeat(inNumRows, inNumCols);
    }

    /// Construct an array by repeating the number of times given by reps.
    template <typename dtype>
    Matrix<dtype> tile(const Matrix<dtype> &inArray, const Shape &inReps)
    {
        return inArray.repeat(inReps);
    }
}
