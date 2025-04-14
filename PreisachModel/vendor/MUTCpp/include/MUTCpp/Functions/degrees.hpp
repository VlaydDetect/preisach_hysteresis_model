// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/rad2deg.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Convert angles from degrees to radians.
    template<typename dtype>
    constexpr auto degrees(dtype inValue) noexcept
    {
        return rad2deg(inValue);
    }
    
    /// Convert angles from degrees to radians.
    template<typename dtype>
    auto degrees(const Matrix<dtype>& inArray)
    {
        return rad2deg(inArray);
    }
}
