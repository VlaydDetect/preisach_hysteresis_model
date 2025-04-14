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
    /// Round to nearest integer towards zero.
    template <typename dtype>
    dtype fix(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue > 0 ? std::floor(inValue) : std::ceil(inValue);
    }

    /// Round to nearest integer towards zero.
\
    template <typename dtype>
    Matrix<dtype> fix(const Matrix<dtype> &inArray)
    {
        Matrix<double> returnArray(inArray.shape());

        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> double { return fix(inValue); });

        return returnArray;
    }
}
