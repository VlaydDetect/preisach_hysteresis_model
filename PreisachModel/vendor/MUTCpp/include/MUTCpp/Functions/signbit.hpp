// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns element-wise True where signbit is set (less than zero).
    template <typename dtype>
    bool signbit(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue < dtype{0};
    }

    /// Returns element-wise True where signbit is set (less than zero).
    template <typename dtype>
    Matrix<bool> signbit(const Matrix<dtype> &inArray)
    {
        Matrix<bool> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> bool { return signbit(inValue); });

        return returnArray;
    }
}
