// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return the identity array.
    ///
    /// The identity array is a square array with ones on the main diagonal.
    template<typename dtype>
    Matrix<dtype> identity(uint32 inSquareSize)
    {
        Matrix<dtype> returnArray(inSquareSize);
        returnArray.zeros();
        for (uint32 i = 0; i < inSquareSize; ++i)
        {
            returnArray(i, i) = dtype{ 1 };
        }

        return returnArray;
    }
}
