// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Types.hpp"
#include "Functions/flip.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Rotate an array by 90 degrees counterclockwise on the plane.
    /// @param inArray
    /// @param inK: the number of times to rotate 90 degrees
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> rot90(const Matrix<dtype> &inArray, uint8 inK = 1)
    {
        inK %= 4;
        switch (inK)
        {
        case 0:
        {
            return inArray;
        }
        case 1:
        {
            return flipud(inArray.transpose());
        }
        case 2:
        {
            return flip(inArray, Axis::NONE);
        }
        case 3:
        {
            return fliplr(inArray.transpose());
        }
        default:
        {
            // this isn't possible, just putting this here to get rid
            // of the compiler warning.
            return {};
        }
        }
    }
}
