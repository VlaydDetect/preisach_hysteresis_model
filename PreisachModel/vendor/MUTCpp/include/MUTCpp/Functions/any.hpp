// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Test whether any array element along a given axis evaluates to True.
    template<typename dtype>
    Matrix<bool> any(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.any(inAxis);
    }
}
