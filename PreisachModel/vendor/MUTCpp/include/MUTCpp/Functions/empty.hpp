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
    /// Return a new empty array.
    template<typename dtype>
    Matrix<dtype> empty()
    {
        return Matrix<dtype>(0, 0);
    }
    
    /// Return a new array of given shape and type, without initializing entries.
    template<typename dtype>
    Matrix<dtype> empty(uint32 inSquareSize)
    {
        return Matrix<dtype>(inSquareSize, inSquareSize);
    }
    
    /// Return a new array of given shape and type, without initializing entries.
    template<typename dtype>
    Matrix<dtype> empty(uint32 inNumRows, uint32 inNumCols)
    {
        return Matrix<dtype>(inNumRows, inNumCols);
    }
    
    /// Return a new array of given shape and type, without initializing entries.
    template<typename dtype>
    Matrix<dtype> empty(const Shape& inShape)
    {
        return Matrix<dtype>(inShape);
    }
}
