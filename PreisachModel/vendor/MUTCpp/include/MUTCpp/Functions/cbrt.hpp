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
    /// Return the cube-root of an array.
    template <typename dtype>
    double cbrt(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::cbrt(static_cast<double>(inValue));
    }

    /// Return the cube-root of an array, element-wise.
    template <typename dtype>
    Matrix<double> cbrt(const Matrix<dtype> &inArray)
    {
        Matrix<double> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> double { return cbrt(inValue); });

        return returnArray;
    }
}
