// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/deg2rad.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Convert angles from degrees to radians.
    template <typename dtype>
    constexpr auto radians(dtype inValue) noexcept
    {
        return deg2rad(inValue);
    }

    /// Convert angles from degrees to radians.
    template <typename dtype>
    auto radians(const Matrix<dtype> &inArray)
    {
        return deg2rad(inArray);
    }
}
