// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Types.hpp"
#include "Functions/diagflat.hpp"
#include "Functions/diagonal.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Extract a diagonal or construct a diagonal array.
    /// @param inArray
    /// @param k Diagonal in question. The default is 0.
    ///     Use k>0 for diagonals above the main diagonal, and k<0
    ///     for diagonals below the main diagonal.
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> diag(const Matrix<dtype>& inArray, int32 k = 0)
    {
        if (inArray.isflat())
        {
            return diagflat(inArray, k);
        }

        return diagonal(inArray, k, Axis::ROW);
    }
}
