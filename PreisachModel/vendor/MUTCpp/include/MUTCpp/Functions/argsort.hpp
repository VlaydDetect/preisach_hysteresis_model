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
    /// Returns the indices that would sort an array.
    template<typename dtype>
    Matrix<uint32> argsort(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.argsort(inAxis);
    }
}
