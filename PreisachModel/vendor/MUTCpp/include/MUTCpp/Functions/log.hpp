// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "Matrix.hpp"

namespace mc
{
    /// Natural logarithm.
    template <typename dtype>
    auto log(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(inValue);
    }

    /// Natural logarithm.
    template <typename dtype>
    auto log(std::complex<dtype> inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(inValue);
    }

    /// Natural logarithm, element-wise.
    template <typename dtype>
    auto log(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(log(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return log(inValue); });

        return returnArray;
    }

    /// Natural logarithm, element-wise.
    template <typename dtype>
    auto log(const Matrix<std::complex<dtype>> &inArray)
    {
        Matrix<decltype(log(std::complex<dtype>{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](std::complex<dtype> inValue) noexcept -> auto { return log(inValue); });

        return returnArray;
    }
}
