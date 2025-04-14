// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/DtypeInfo.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Test for NaN and return result as a boolean.
    template <typename dtype>
    bool isnan(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (DtypeInfo<dtype>::isInteger())
        {
            return false;
        }

        return std::isnan(static_cast<double>(inValue)); // static_cast is needed for compiler error
    }

    //============================================================================
    // Method Description:
    /// Test element-wise for NaN and return result as a boolean array.
    template <typename dtype>
    Matrix<bool> isnan(const Matrix<dtype> &inArray)
    {
        Matrix<bool> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> bool { return isnan(inValue); });

        return returnArray;
    }
}
