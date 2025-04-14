// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Compute the median along the specified axis.
    template<typename dtype>
    Matrix<dtype> median(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.median(inAxis);
    }
}
