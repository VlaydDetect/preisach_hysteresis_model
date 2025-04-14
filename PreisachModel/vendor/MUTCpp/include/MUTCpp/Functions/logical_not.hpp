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
    /// Compute the truth value of NOT x element-wise.
    template <typename dtype>
    Matrix<bool> logical_not(const Matrix<dtype> &inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) -> bool { return utils::essentiallyEqual(inValue, dtype{0}); });

        return returnArray;
    }
}
