// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Constants.hpp"
#include "Core/Shape.hpp"
#include "Functions/full.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    inline Matrix<double> nans(uint32 inSquareSize)
    {
        return full(inSquareSize, consts::nan);
    }
    
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    inline Matrix<double> nans(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, consts::nan);
    }
    
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    inline Matrix<double> nans(const Shape& inShape)
    {
        return full(inShape, consts::nan);
    }
}
