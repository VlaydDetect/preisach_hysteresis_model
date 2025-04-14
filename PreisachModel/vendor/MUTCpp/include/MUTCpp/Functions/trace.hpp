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
    /// Return the sum along diagonals of the array.
    /// @param inArray
    /// @param inOffset: (Offset from main diagonal, default = 0, negative=above, positive=below)
    /// @param inAxis (Optional, default ROW)
    /// @return Matrix
    template <typename dtype>
    dtype trace(const Matrix<dtype> &inArray, int16 inOffset = 0, Axis inAxis = Axis::ROW) noexcept
    {
        return inArray.trace(inOffset, inAxis);
    }
}
