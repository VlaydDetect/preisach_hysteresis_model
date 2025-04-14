// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the truth value of x1 XOR x2 element-wise.
    template <typename dtype>
    Matrix<bool> logical_xor(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster<bool>(inArray1,
                                            inArray2,
                                            [](dtype inValue1, dtype inValue2) -> bool
                                            {
                                                return !utils::essentiallyEqual(inValue1, dtype{0}) !=
                                                    !utils::essentiallyEqual(inValue2, dtype{0});
                                            });
    }
}
