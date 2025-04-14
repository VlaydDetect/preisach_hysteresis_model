// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/full.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a new array of given shape and type, filled with ones.
    template<typename dtype>
    Matrix<dtype> ones(uint32 inSquareSize)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return full(inSquareSize, inSquareSize, dtype{ 1 });
    }

    /// Return a new array of given shape and type, filled with ones.
    template<typename dtype>
    Matrix<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return full(inNumRows, inNumCols, dtype{ 1 });
    }

    /// Return a new array of given shape and type, filled with ones.
    template<typename dtype>
    Matrix<dtype> ones(const Shape& inShape)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return full(inShape, dtype{ 1 });
    }
}
