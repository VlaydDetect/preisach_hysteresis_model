// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/TypeTraits.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the remainder o the division.
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept
    {
        return inValue1 % inValue2;
    }

    /// Return the remainder of the division.
    template<typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int> = 0>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept
    {
        return std::fmod(inValue1, inValue2);
    }

    /// Return the element-wise remainder of the division.
    template<typename dtype>
    Matrix<dtype> fmod(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        return inArray1 % inArray2;
    }
}
