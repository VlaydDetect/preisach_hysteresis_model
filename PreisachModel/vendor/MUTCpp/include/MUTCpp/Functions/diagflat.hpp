// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Create a two-dimensional array with the flattened input as a diagonal.
    /// @param inArray
    /// @param k Diagonal to set; 0, the default, corresponds to the �main� diagonal,
    /// a positive (negative) k giving the number of the diagonal above (below) the main.
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> diagflat(const Matrix<dtype> &inArray, int32 k = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto absK = static_cast<uint32>(std::abs(k));
        Matrix<dtype> returnArray(inArray.size() + absK);

        const uint32 rowOffset = k < 0 ? absK : 0;
        const uint32 colOffset = k > 0 ? absK : 0;

        returnArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            returnArray(i + rowOffset, i + colOffset) = inArray[i];
        }

        return returnArray;
    }
}
