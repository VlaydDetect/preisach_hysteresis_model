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
    /// Find the indices of array elements that are non-zero, grouped by element.
    template<typename dtype>
    Matrix<uint32> argwhere(const Matrix<dtype>& inArray)
    {
        return inArray.flatnonzero();
    }
}
