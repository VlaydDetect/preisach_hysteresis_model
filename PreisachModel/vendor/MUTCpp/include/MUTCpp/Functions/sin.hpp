// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Trigonometric sine.
    template <typename dtype>
    auto sin(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::sin(inValue);
    }

    /// Trigonometric sine, element-wise.
    template <typename dtype>
    auto sin(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(sin(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return sin(inValue); });

        return returnArray;
    }
}
