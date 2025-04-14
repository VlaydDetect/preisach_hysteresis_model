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
    /// Returns the indices of the minimum values along an axis.
    template<typename dtype>
    Matrix<uint32> argmin(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.argmin(inAxis);
    }
}
