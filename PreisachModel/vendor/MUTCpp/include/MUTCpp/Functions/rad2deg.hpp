// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Constants.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Convert angles from radians to degrees.
    template <typename dtype>
    constexpr auto rad2deg(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue * 180. / consts::pi;
    }

    /// Convert angles from radians to degrees.
    template <typename dtype>
    auto rad2deg(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(rad2deg(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return rad2deg(inValue); });

        return returnArray;
    }
}
