// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Test for inf and return result as a boolean.
    template <typename dtype>
    bool isinf(dtype inValue) noexcept
    {
        STATIC_ASSERT_FLOAT(dtype);

        return std::isinf(inValue);
    }

    /// Test element-wise for inf and return result as a boolean array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isinf.html
    ///
    /// @param inArray
    ///
    /// @return Matrix
    ///
    template <typename dtype>
    Matrix<bool> isinf(const Matrix<dtype> &inArray)
    {
        Matrix<bool> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> bool { return isinf(inValue); });

        return returnArray;
    }
}
