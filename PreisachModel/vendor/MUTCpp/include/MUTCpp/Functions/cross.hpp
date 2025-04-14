// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the cross-product of two (arrays of) vectors.
    template <typename dtype>
    Matrix<dtype> cross(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("the input array dimensions are not consistant.");
        }

        switch (inAxis)
        {
        case Axis::NONE:
        {
            const uint32 arraySize = inArray1.size();
            if (arraySize != inArray2.size() || arraySize < 2 || arraySize > 3)
            {
                THROW_INVALID_ARGUMENT_ERROR(
                    "incompatible dimensions for cross product (dimension must be 2 or 3)");
            }

            Matrix<dtype> in1 = inArray1.flatten();
            Matrix<dtype> in2 = inArray2.flatten();

            switch (arraySize)
            {
            case 2:
            {
                Matrix<dtype> returnArray = {in1[0] * in2[1] - in1[1] * in2[0]};
                return returnArray;
            }
            case 3:
            {
                dtype i = in1[1] * in2[2] - in1[2] * in2[1];
                dtype j = -(in1[0] * in2[2] - in1[2] * in2[0]);
                dtype k = in1[0] * in2[1] - in1[1] * in2[0];

                Matrix<dtype> returnArray = {i, j, k};
                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented array size.");
                return {}; // get rid of compiler warning
            }
            }
        }
        case Axis::ROW:
        {
            const Shape arrayShape = inArray1.shape();
            if (arrayShape != inArray2.shape() || arrayShape.m_Rows < 2 || arrayShape.m_Rows > 3)
            {
                THROW_INVALID_ARGUMENT_ERROR(
                    "incompatible dimensions for cross product (dimension must be 2 or 3)");
            }

            Shape returnArrayShape;
            returnArrayShape.m_Cols = arrayShape.m_Cols;
            if (arrayShape.m_Rows == 2)
            {
                returnArrayShape.m_Rows = 1;
            }
            else
            {
                returnArrayShape.m_Rows = 3;
            }

            Matrix<dtype> returnArray(returnArrayShape);
            for (uint32 col = 0; col < arrayShape.m_Cols; ++col)
            {
                const auto theCol = static_cast<int32>(col);
                Matrix<dtype> vec1 = inArray1(inArray1.rSlice(), {theCol, theCol + 1});
                Matrix<dtype> vec2 = inArray2(inArray2.rSlice(), {theCol, theCol + 1});
                Matrix<dtype> vecCross = cross(vec1, vec2, Axis::NONE);

                returnArray.put({0, static_cast<int32>(returnArrayShape.m_Rows)}, {theCol, theCol + 1}, vecCross);
            }

            return returnArray;
        }
        case Axis::COL:
        {
            const Shape arrayShape = inArray1.shape();
            if (arrayShape != inArray2.shape() || arrayShape.m_Cols < 2 || arrayShape.m_Cols > 3)
            {
                THROW_INVALID_ARGUMENT_ERROR(
                    "incompatible dimensions for cross product (dimension must be 2 or 3)");
            }

            Shape returnArrayShape;
            returnArrayShape.m_Rows = arrayShape.m_Rows;
            if (arrayShape.m_Cols == 2)
            {
                returnArrayShape.m_Cols = 1;
            }
            else
            {
                returnArrayShape.m_Cols = 3;
            }

            Matrix<dtype> returnArray(returnArrayShape);
            for (uint32 row = 0; row < arrayShape.m_Rows; ++row)
            {
                const auto theRow = static_cast<int32>(row);
                Matrix<dtype> vec1 = inArray1({theRow, theRow + 1}, inArray1.cSlice());
                Matrix<dtype> vec2 = inArray2({theRow, theRow + 1}, inArray2.cSlice());
                Matrix<dtype> vecCross = cross(vec1, vec2, Axis::NONE);

                returnArray.put({theRow, theRow + 1}, {0, static_cast<int32>(returnArrayShape.m_Cols)}, vecCross);
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
