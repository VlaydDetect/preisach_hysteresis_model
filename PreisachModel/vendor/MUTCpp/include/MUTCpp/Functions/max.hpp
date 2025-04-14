// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return the maximum of to values;
    template<typename dtype>
    dtype max(const dtype a, const dtype b)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);
        return std::max(a, b);
    }
    
    /// Return the maximum of an array or maximum along an axis.
    template<typename dtype>
    Matrix<dtype> max(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.max(inAxis);
    }
}
