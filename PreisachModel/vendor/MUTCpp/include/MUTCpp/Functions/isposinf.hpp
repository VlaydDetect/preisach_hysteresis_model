// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Functions/isinf.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Test for negative inf and return result as a boolean.
    template <typename dtype>
    bool isposinf(dtype inValue) noexcept
    {
        STATIC_ASSERT_FLOAT(dtype);

        return inValue > 0 && std::isinf(inValue);
    }

    /// Test element-wise for negative inf and return result as a boolean array.
    template <typename dtype>
    Matrix<bool> isposinf(const Matrix<dtype> &inArray)
    {
        Matrix<bool> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> bool { return isposinf(inValue); });

        return returnArray;
    }
}
