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
    /// Return the natural logarithm of one plus the input array.
    /// Calculates log(1 + x).
    template <typename dtype>
    auto log1p(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::log1p(inValue);
    }

    /// Return the natural logarithm of one plus the input array, element-wise.
    /// Calculates log(1 + x).
    template <typename dtype>
    auto log1p(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(log1p(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return log1p(inValue); });

        return returnArray;
    }
}
