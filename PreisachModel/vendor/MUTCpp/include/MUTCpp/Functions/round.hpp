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
    /// Round value to the given number of decimals.
    template<typename dtype>
    dtype round(dtype inValue, uint8 inDecimals = 0)
    {
        Matrix<dtype> input = { inValue };
        return input.round(inDecimals).item();
    }

    /// Round an array to the given number of decimals.
    template<typename dtype>
    Matrix<dtype> round(const Matrix<dtype>& inArray, uint8 inDecimals = 0)
    {
        return inArray.round(inDecimals);
    }
}
