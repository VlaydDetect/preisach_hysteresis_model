// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Change the sign of x1 to that of x2, element-wise.
    template <typename dtype>
    Matrix<dtype> copySign(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input arrays are not consistant.");
        }

        Matrix<dtype> returnArray(inArray1.shape());
        algo::transform(inArray1.cbegin(),
                        inArray1.cend(),
                        inArray2.cbegin(),
                        returnArray.begin(),
                        [](dtype inValue1, dtype inValue2) -> dtype
                        {
                            return inValue2 < dtype{0} ? std::abs(inValue1) * -1 : std::abs(inValue1);
                        });

        return returnArray;
    }
}
