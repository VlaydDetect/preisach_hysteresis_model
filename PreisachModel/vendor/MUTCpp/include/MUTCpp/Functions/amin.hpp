// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return the minimum of an array or minimum along an axis.
    template<typename dtype>
    Matrix<dtype> amin(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.min(inAxis);
    }
}
