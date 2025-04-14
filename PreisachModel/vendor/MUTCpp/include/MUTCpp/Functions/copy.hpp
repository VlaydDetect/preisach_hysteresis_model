// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Return an array copy of the given object.
    template<typename dtype>
    Matrix<dtype> copy(const Matrix<dtype>& inArray)
    {
        return Matrix<dtype>(inArray);
    }
}
