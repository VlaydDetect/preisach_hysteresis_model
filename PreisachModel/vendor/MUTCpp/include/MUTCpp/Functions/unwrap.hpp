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
    /// Unwrap by changing deltas between values to 2*pi complement.
    /// Unwraps to [-pi, pi].
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    template <typename dtype>
    dtype unwrap(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return static_cast<dtype>(std::atan2(std::sin(inValue), std::cos(inValue)));
    }

    /// Unwrap by changing deltas between values to 2*pi complement.
    /// Unwraps to [-pi, pi].
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    template <typename dtype>
    Matrix<dtype> unwrap(const Matrix<dtype> &inArray)
    {
        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> dtype { return unwrap(inValue); });

        return returnArray;
    }
}
