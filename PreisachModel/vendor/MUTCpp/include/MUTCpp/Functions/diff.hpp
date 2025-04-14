// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Calculate the n-th discrete difference along the given axis.
    /// Unsigned dtypes will give you weird results...obviously.
    template <typename dtype>
    Matrix<dtype> diff(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();

        switch (inAxis)
        {
        case Axis::NONE:
        {
            if (inArray.size() < 2)
            {
                return Matrix<dtype>(0);
            }

            Matrix<dtype> returnArray(1, inArray.size() - 1);
            algo::transform(inArray.cbegin(),
                            inArray.cend() - 1,
                            inArray.cbegin() + 1,
                            returnArray.begin(),
                            [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 - inValue1; });

            return returnArray;
        }
        case Axis::COL:
        {
            if (inShape.m_Cols < 2)
            {
                return Matrix<dtype>(0);
            }

            Matrix<dtype> returnArray(inShape.m_Rows, inShape.m_Cols - 1);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                algo::transform(inArray.cbegin(row),
                                inArray.cend(row) - 1,
                                inArray.cbegin(row) + 1,
                                returnArray.begin(row),
                                [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 - inValue1; });
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return diff(inArray.transpose(), Axis::COL).transpose();
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
