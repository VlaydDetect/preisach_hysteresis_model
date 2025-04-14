// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include "Matrix.hpp"

namespace mc
{
    /// Evenly round to the given number of decimals.
    template<typename dtype>
    dtype around(dtype inValue, uint8 inNumDecimals = 0)
    {
        Matrix<dtype> value = { inValue };
        return value.round(inNumDecimals).item();
    }

    //============================================================================
    // Method Description:
    /// Evenly round to the given number of decimals.
    template<typename dtype>
    Matrix<dtype> around(const Matrix<dtype>& inArray, uint8 inNumDecimals = 0)
    {
        return inArray.round(inNumDecimals);
    }
}
