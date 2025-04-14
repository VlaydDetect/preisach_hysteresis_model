// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Reverse the order of elements in an array along the given axis.
    template<typename dtype>
    Matrix<dtype> flip(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
        case Axis::NONE:
        {
            Matrix<dtype> returnArray(inArray);
            algo::reverse(returnArray.begin(), returnArray.end());
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<dtype> returnArray(inArray);
            for (uint32 row = 0; row < inArray.shape().rows; ++row)
            {
                algo::reverse(returnArray.begin(row), returnArray.end(row));
            }
            return returnArray;
        }
        case Axis::ROW:
        {
            return flip(inArray.transpose(), Axis::COL).transpose();
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }

    /// Flip array in the left/right direction.
    template<typename dtype>
    Matrix<dtype> fliplr(const Matrix<dtype>& inArray)
    {
        return flip(inArray, Axis::COL);
    }

    /// Flip array in the up/down direction.
    template<typename dtype>
    Matrix<dtype> flipud(const Matrix<dtype>& inArray)
    {
        return flip(inArray, Axis::ROW);
    }
}
