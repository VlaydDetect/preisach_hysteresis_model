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
    /// Counts the number of non-zero values in the array.
    template <typename dtype>
    Matrix<uint32> count_nonzero(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            Matrix<uint32> count = {inArray.size() -
                static_cast<uint32>(
                    algo::count(inArray.cbegin(), inArray.cend(), dtype{0}))};
            return count;
        }
        case Axis::COL:
        {
            Shape inShape = inArray.shape();

            Matrix<uint32> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                returnArray(0, row) =
                    inShape.m_Cols - static_cast<uint32>(algo::count(inArray.cbegin(row), inArray.cend(row), dtype{0}));
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return count_nonzero(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
