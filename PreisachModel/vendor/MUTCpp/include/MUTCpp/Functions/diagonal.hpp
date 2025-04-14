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
    /// Return specified diagonals.
    /// @param inArray
    /// @param inOffset (Defaults to 0)
    /// @param inAxis (Optional, default ROW) axis the offset is applied to
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> diagonal(const Matrix<dtype> &inArray, int32 inOffset = 0, Axis inAxis = Axis::ROW)
    {
        return inArray.diagonal(inOffset, inAxis);
    }
}
