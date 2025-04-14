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
    /// Calculate 2**p for all p in the input value.
    template<typename dtype>
    auto exp2(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::exp2(inValue);
    }
    
    /// Calculate 2**p for all p in the input array.
    template<typename dtype>
    auto exp2(const Matrix<dtype>& inArray)
    {
        Matrix<decltype(exp(dtype{ 0 }))> returnArray(inArray.shape());

        algo::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [](dtype inValue) noexcept -> auto { return exp2(inValue); });

        return returnArray;
    }
}
