// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return the cumulative sum of the elements along a given axis.
    template<typename dtype>
    Matrix<dtype> cumsum(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.cumsum(inAxis);
    }
}
