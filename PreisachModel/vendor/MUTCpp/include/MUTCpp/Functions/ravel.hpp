// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Flattens the array but does not make a copy.
    template<typename dtype>
    Matrix<dtype>& ravel(Matrix<dtype>& inArray) noexcept
    {
        inArray.ravel();
        return inArray;
    }
}
