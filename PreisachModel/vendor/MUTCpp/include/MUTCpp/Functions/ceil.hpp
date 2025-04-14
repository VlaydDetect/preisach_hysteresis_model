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
    /// Return the ceiling of the input.
    template<typename dtype>
    dtype ceil(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::ceil(inValue);
    }

    /// Return the ceiling of the input, element-wise.
    template<typename dtype>
    Matrix<dtype> ceil(const Matrix<dtype>& inArray)
    {
        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [](dtype inValue) noexcept -> dtype { return ceil(inValue); });

        return returnArray;
    }
}
