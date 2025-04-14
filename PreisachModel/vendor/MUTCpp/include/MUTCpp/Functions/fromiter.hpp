// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <iterator>
#include <vector>

#include "Matrix.hpp"

namespace mc
{
    /// Create a new 1-dimensional array from an iterable object.
    template<typename dtype, typename Iter>
    Matrix<dtype> fromiter(Iter inBegin, Iter inEnd)
    {
        return Matrix<dtype>(inBegin, inEnd);
    }
}
