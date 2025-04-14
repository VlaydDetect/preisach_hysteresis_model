// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/array_equiv.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// True if two arrays have the same shape and elements, False otherwise.
    template<typename dtype>
    bool array_equal(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2) noexcept
    {
        if (inArray1.shape() != inArray2.shape())
        {
            return false;
        }

        return array_equiv(inArray1, inArray2);
    }
}
