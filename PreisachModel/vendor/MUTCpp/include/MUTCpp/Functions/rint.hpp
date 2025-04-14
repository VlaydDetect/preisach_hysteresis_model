// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Round value to the nearest integer.
    template <typename dtype>
    dtype rint(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::rint(inValue);
    }

    /// Round elements of the array to the nearest integer.
    template <typename dtype>
    Matrix<dtype> rint(const Matrix<dtype> &inArray)
    {
        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> dtype { return rint(inValue); });

        return returnArray;
    }
}
