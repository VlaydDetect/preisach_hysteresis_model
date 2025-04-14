// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>

#include "Core/StaticAssert.hpp"
#include "Core/ComplexOperators.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Clip (limit) the value.
    template<typename dtype>
    dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

#ifdef __cpp_lib_clamp
        const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

        return std::clamp(inValue, inMinValue, inMaxValue, comparitor);
#else
        if (inValue < inMinValue)
        {
            return inMinValue;
        }
        else if (inValue > inMaxValue)
        {
            return inMaxValue;
        }

        return inValue;
#endif
    }
    
    /// Clip (limit) the values in an array.
    template<typename dtype>
    Matrix<dtype> clip(const Matrix<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return inArray.clip(inMinValue, inMaxValue);
    }
}
