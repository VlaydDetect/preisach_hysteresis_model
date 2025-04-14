// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Cubes the input
    template <typename dtype>
    constexpr dtype cube(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return utils::cube(inValue);
    }

    /// Cubes the elements of the array
    template <typename dtype>
    Matrix<dtype> cube(const Matrix<dtype> &inArray)
    {
        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> dtype { return cube(inValue); });

        return returnArray;
    }
}
