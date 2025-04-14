// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return a new array of given shape and type, filled with nans.
    template<typename dtype>
    Matrix<double> nans_like(const Matrix<dtype>& inArray)
    {
        Matrix<double> returnArray(inArray.shape());
        returnArray.nans();
        return returnArray;
    }
}
