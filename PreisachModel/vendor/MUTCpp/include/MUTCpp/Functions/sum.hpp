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
    /// Sum of array elements over a given axis.
    template <typename dtype>
    Matrix<dtype> sum(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.sum(inAxis);
    }
}
