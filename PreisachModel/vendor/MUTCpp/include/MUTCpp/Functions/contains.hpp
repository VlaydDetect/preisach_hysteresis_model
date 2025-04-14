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
    /// returns whether a value is included the array
    template<typename dtype>
    Matrix<bool> contains(const Matrix<dtype>& inArray, dtype inValue, Axis inAxis = Axis::NONE)
    {
        return inArray.contains(inValue, inAxis);
    }
}
