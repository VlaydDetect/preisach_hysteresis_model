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
    /// An array with ones at and below the given diagonal and zeros elsewhere.
    /// @param inN: number of rows and cols
    /// @param inOffset: (the sub-diagonal at and below which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> tril(uint32 inN, int32 inOffset = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else
        {
            rowStart = inOffset * -1;
        }

        Matrix<dtype> returnArray(inN);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inN)
                {
                    break;
                }

                returnArray(row, col) = dtype{1};
            }
        }

        return returnArray;
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    /// @param inN: number of rows
    /// @param inM: number of columns
    /// @param inOffset: (the sub-diagonal at and below which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> tril(uint32 inN, uint32 inM, int32 inOffset = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        Matrix<dtype> returnArray(inN, inM);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = dtype{1};
            }
        }

        return returnArray;
    }

    // forward declare
    template <typename dtype>
    Matrix<dtype> triu(uint32 inN, uint32 inM, int32 inOffset = 0);

    /// Lower triangle of an array.
    /// Return a copy of an array with elements above the k - th diagonal zeroed.
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tril.html
    /// @param inArray: number of rows and cols
    /// @param inOffset: (the sub-diagonal at and below which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> tril(const Matrix<dtype> &inArray, int32 inOffset = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(triu<bool>(inShape.m_Rows, inShape.m_Cols, inOffset + 1), 0);
        return outArray;
    }

    /// An array with ones at and above the given diagonal and zeros elsewhere.
    /// @param inN: number of rows
    /// @param inM: number of columns
    /// @param inOffset: (the sub-diagonal at and above which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> triu(uint32 inN, uint32 inM, int32 inOffset)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        // because i'm stealing the lines of code from tril and reversing it, this is necessary
        inOffset -= 1;

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        Matrix<dtype> returnArray(inN, inM);
        returnArray.ones();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = dtype{0};
            }
        }

        return returnArray;
    }

    /// An array with ones at and above the given diagonal and zeros elsewhere.
    /// @param inN: number of rows and cols
    /// @param inOffset: (the sub-diagonal at and above which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> triu(uint32 inN, int32 inOffset = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return tril<dtype>(inN, -inOffset).transpose();
    }

    /// Upper triangle of an array.
    /// Return a copy of an array with elements below the k - th diagonal zeroed.
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.triu.html
    /// @param inArray: number of rows and cols
    /// @param inOffset: (the sub-diagonal at and below which the array is filled.
    /// k = 0 is the main diagonal, while k < 0 is below it,
    /// and k > 0 is above. The default is 0.)
    template <typename dtype>
    Matrix<dtype> triu(const Matrix<dtype> &inArray, int32 inOffset = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(tril<bool>(inShape.m_Rows, inShape.m_Cols, inOffset - 1), 0);
        return outArray;
    }
}
