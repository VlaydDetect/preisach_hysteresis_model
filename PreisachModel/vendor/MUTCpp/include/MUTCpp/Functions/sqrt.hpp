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
    /// Return the positive square-root of a value.
    template <typename dtype>
    auto sqrt(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::sqrt(inValue);
    }

    /// Return the positive square-root of an array, element-wise.
    template <typename dtype>
    auto sqrt(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(sqrt(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return sqrt(inValue); });

        return returnArray;
    }
}
