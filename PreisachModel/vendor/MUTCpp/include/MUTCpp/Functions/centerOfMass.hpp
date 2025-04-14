// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns the center of mass of the array values along an axis.
    /// @param inArray
    /// @param inAxis (Optional, default NONE which is a 2d center of mass)
    /// @return Matrix: if axis is NONE, then a 1x2 array of the centroid row/col is returned.
    template <typename dtype>
    Matrix<double> centerOfMass(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const Shape shape = inArray.shape();

        switch (inAxis)
        {
        case Axis::NONE:
        {
            double inten = 0.;
            double rowCenter = 0.;
            double colCenter = 0.;

            for (uint32 row = 0; row < shape.m_Rows; ++row)
            {
                for (uint32 col = 0; col < shape.m_Cols; ++col)
                {
                    const auto pixelValue = static_cast<double>(inArray(row, col));

                    inten += pixelValue;
                    rowCenter += pixelValue * static_cast<double>(row);
                    colCenter += pixelValue * static_cast<double>(col);
                }
            }

            rowCenter /= inten;
            colCenter /= inten;

            return {rowCenter, colCenter};
        }
        case Axis::ROW:
        {
            Matrix<double> returnArray(1, shape.m_Cols);
            returnArray.zeros();

            const Matrix<double> inten = inArray.template astype<double>().sum(inAxis);

            for (uint32 colIdx = 0; colIdx < shape.m_Cols; ++colIdx)
            {
                for (uint32 rowIdx = 0; rowIdx < shape.m_Rows; ++rowIdx)
                {
                    returnArray(0, colIdx) +=
                        static_cast<double>(inArray(rowIdx, colIdx)) * static_cast<double>(rowIdx);
                }

                returnArray(0, colIdx) /= inten[colIdx];
            }

            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> returnArray(1, shape.m_Rows);
            returnArray.zeros();

            const Matrix<double> inten = inArray.template astype<double>().sum(inAxis);

            for (uint32 rowIdx = 0; rowIdx < shape.m_Rows; ++rowIdx)
            {
                for (uint32 colIdx = 0; colIdx < shape.m_Cols; ++colIdx)
                {
                    returnArray(0, rowIdx) +=
                        static_cast<double>(inArray(rowIdx, colIdx)) * static_cast<double>(colIdx);
                }

                returnArray(0, rowIdx) /= inten[rowIdx];
            }

            return returnArray;
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
