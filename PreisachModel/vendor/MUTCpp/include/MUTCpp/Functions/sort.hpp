// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    //============================================================================
    // Method Description:
    /// Return a sorted copy of an array.
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> sort(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        Matrix<dtype> returnArray(inArray);
        returnArray.sort(inAxis);
        return returnArray;
    }
}

