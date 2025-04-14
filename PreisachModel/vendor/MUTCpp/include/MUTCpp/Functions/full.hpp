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
    /// Return a new array of given shape and type, filled with inFillValue
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> full(uint32 inSquareSize, dtype inFillValue)
    {
        Matrix<dtype> returnArray(inSquareSize, inSquareSize);
        returnArray.fill(inFillValue);
        return returnArray;
    }
    
    /// Return a new array of given shape and type, filled with inFillValue
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
    {
        Matrix<dtype> returnArray(inNumRows, inNumCols);
        returnArray.fill(inFillValue);
        return returnArray;
    }
    
    /// Return a new array of given shape and type, filled with inFillValue
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> full(const Shape& inShape, dtype inFillValue)
    {
        return full(inShape.m_Rows, inShape.m_Cols, inFillValue);
    }
}
