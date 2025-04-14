// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Interchange two axes of an array.
    template<typename dtype>
    Matrix<dtype> swapaxes(const Matrix<dtype>& inArray)
    {
        return inArray.swapaxes();
    }
}
