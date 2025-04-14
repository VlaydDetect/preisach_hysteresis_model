// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If a single integer,
    /// then the result will be a 1-D array of that length. One shape dimension
    /// can be -1. In this case, the value is inferred from the length of the
    /// array and remaining dimensions.
    template<typename dtype>
    Matrix<dtype>& reshape(Matrix<dtype>& inArray, uint32 inSize)
    {
        inArray.reshape(inSize);
        return inArray;
    }

    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If a single integer,
    /// then the result will be a 1-D array of that length. One shape dimension
    /// can be -1. In this case, the value is inferred from the length of the
    /// array and remaining dimensions.
    template<typename dtype>
    Matrix<dtype>& reshape(Matrix<dtype>& inArray, int32 inNumRows, int32 inNumCols)
    {
        inArray.reshape(inNumRows, inNumCols);
        return inArray;
    }

    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If a single integer,
    /// then the result will be a 1-D array of that length. One shape dimension
    /// can be -1. In this case, the value is inferred from the length of the
    /// array and remaining dimensions.
    template<typename dtype>
    Matrix<dtype>& reshape(Matrix<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.reshape(inNewShape);
        return inArray;
    }
}
