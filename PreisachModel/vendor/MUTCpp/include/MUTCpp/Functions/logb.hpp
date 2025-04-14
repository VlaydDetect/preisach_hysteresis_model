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
    /// Logarithm of an arbitrary base
    ///
    /// @param inValue
    /// @param inBase: the logarithm base
    ///
    /// @return value
    template <typename dtype>
    auto logb(dtype inValue, dtype inBase) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(inValue) / std::log(inBase);
    }

    /// Logarithm of an arbitrary base
    ///
    /// @param inArray
    /// @param inBase: the logorithm base
    ///
    /// @return Matrix
    template <typename dtype>
    auto logb(const Matrix<dtype> &inArray, dtype inBase)
    {
        Matrix<decltype(logb(dtype{0}, dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [inBase](dtype inValue) noexcept -> auto { return logb(inValue, inBase); });

        return returnArray;
    }
}
