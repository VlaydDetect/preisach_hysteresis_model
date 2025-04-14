// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/StaticAssert.hpp"
#include "Functions/floor.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the largest integer smaller or equal to the division of the inputs.
    template <typename dtype>
    dtype floor_divide(dtype inValue1, dtype inValue2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::floor(inValue1 / inValue2);
    }

    /// Return the largest integer smaller or equal to the division of the inputs.
    template <typename dtype>
    Matrix<dtype> floor_divide(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        return floor(inArray1 / inArray2);
    }
}
