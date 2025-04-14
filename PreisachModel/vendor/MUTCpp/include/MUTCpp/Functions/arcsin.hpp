// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "MUTCpp/Core/StaticAssert.hpp"
#include "MUTCpp/Core/Algorithm.hpp"
#include "MUTCpp/Matrix.hpp"

namespace mc
{
    /// Trigonometric inverse sine.
    template <typename dtype>
    auto arcsin(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::asin(inValue);
    }

    /// Trigonometric inverse sine, element-wise.
    template <typename dtype>
    auto arcsin(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(arcsin(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return arcsin(inValue); });

        return returnArray;
    }
}
