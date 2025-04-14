// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/Types.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns an element-wise indication of the sign of a number.
    /// The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    /// nan is returned for nan inputs.
    template <typename dtype>
    int8 sign(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inValue < dtype{0})
        {
            return -1;
        }

        if (inValue > dtype{0})
        {
            return 1;
        }

        return 0;
    }

    /// Returns an element-wise indication of the sign of a number.
    /// The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    /// nan is returned for nan inputs.
    template <typename dtype>
    Matrix<int8> sign(const Matrix<dtype> &inArray)
    {
        Matrix<int8> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> int8 { return sign(inValue); });

        return returnArray;
    }
}
