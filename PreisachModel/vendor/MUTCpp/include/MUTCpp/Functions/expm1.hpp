// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Calculate exp(x) - 1 for the input value.
    template<typename dtype>
    auto expm1(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::exp(inValue) - dtype{ 1 };
    }
    
    /// Calculate exp(x) - 1 for all elements in the array.
    template<typename dtype>
    auto expm1(const Matrix<dtype>& inArray)
    {
        Matrix<decltype(exp(dtype{ 0 }))> returnArray(inArray.shape());

        algo::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [](dtype inValue) noexcept -> auto { return expm1(inValue); });

        return returnArray;
    }
}
