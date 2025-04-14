// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Replaces the matching elements of an array with the new value
    /// @param inArray
    /// @param oldValue: the value to replace
    /// @param newValue: the value to replace with
    /// @return NdArray
    template <typename dtype>
    Matrix<dtype> replace(const Matrix<dtype> &inArray, dtype oldValue, dtype newValue)
    {
        auto returnArray = inArray.copy();
        returnArray.replace(oldValue, newValue);
        return returnArray;
    }
}
