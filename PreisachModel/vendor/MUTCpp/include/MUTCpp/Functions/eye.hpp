// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    /// @param inN: number of rows (N)
    /// @param inM: number of columns (M)
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> eye(uint32 inN, uint32 inM, int32 inK = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<dtype> returnArray(inN, inM);
        returnArray.zeros();

        if (inK < 0)
        {
            uint32 col = 0;
            for (uint32 row = inK; row < inN; ++row)
            {
                if (col >= inM)
                {
                    break;
                }

                returnArray(row, col++) = dtype{ 1 };
            }
        }
        else
        {
            uint32 row = 0;
            for (uint32 col = inK; col < inM; ++col)
            {
                if (row >= inN)
                {
                    break;
                }

                returnArray(row++, col) = dtype{ 1 };
            }
        }

        return returnArray;
    }
    
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    /// @param inN: number of rows and columns (N)
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> eye(uint32 inN, int32 inK = 0)
    {
        return eye<dtype>(inN, inN, inK);
    }
    
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    /// @param inShape
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> eye(const Shape& inShape, int32 inK = 0)
    {
        return eye<dtype>(inShape.m_Rows, inShape.m_Cols, inK);
    }
}
