// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <utility>

#include "Core/StaticAssert.hpp"
#include "Functions/arange.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return coordinate matrices from coordinate vectors.
    /// Make 2D coordinate arrays for vectorized evaluations of 2D scalar
    /// vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
    /// If input arrays are not one dimensional, they will be flattened.
    /// @return std::pair<Matrix<dtype>, Matrix<dtype> >, i and j matrices
    template <typename dtype>
    std::pair<Matrix<dtype>, Matrix<dtype>> meshgrid(const Matrix<dtype> &inICoords, const Matrix<dtype> &inJCoords)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const uint32 numRows = inJCoords.size();
        const uint32 numCols = inICoords.size();
        auto returnArrayI = Matrix<dtype>(numRows, numCols);
        auto returnArrayJ = Matrix<dtype>(numRows, numCols);

        // first the I array
        for (uint32 row = 0; row < numRows; ++row)
        {
            for (uint32 col = 0; col < numCols; ++col)
            {
                returnArrayI(row, col) = inICoords[col];
            }
        }

        // then the I array
        for (uint32 col = 0; col < numCols; ++col)
        {
            for (uint32 row = 0; row < numRows; ++row)
            {
                returnArrayJ(row, col) = inJCoords[row];
            }
        }

        return std::make_pair(returnArrayI, returnArrayJ);
    }

    /// Return coordinate matrices from coordinate vectors.
    /// Make 2D coordinate arrays for vectorized evaluations of 2D scalar
    /// vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
    /// @return std::pair<Matrix<dtype>, Matrix<dtype> >, i and j matrices
    template <typename dtype>
    std::pair<Matrix<dtype>, Matrix<dtype>> meshgrid(const Slice &inSlice1, const Slice &inSlice2)
    {
        return meshgrid(arange<dtype>(inSlice1), arange<dtype>(inSlice2));
    }
}
