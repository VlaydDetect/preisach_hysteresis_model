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
    /// Convert angles from degrees to radians.
    template <typename dtype>
    constexpr auto deg2rad(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return inValue * consts::pi / 180.;
    }

    /// Convert angles from degrees to radians.
    template <typename dtype>
    auto deg2rad(const Matrix<dtype> &inArray)
    {
        Matrix<decltype(deg2rad(dtype{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> auto { return deg2rad(inValue); });

        return returnArray;
    }
}
