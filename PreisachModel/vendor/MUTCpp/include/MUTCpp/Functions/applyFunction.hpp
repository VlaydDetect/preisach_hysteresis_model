// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>

#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Apply the input function element wise to the input
    /// array in place.
    template<typename dtype>
    void applyFunction(Matrix<dtype>& inArray, const std::function<dtype(dtype)>& inFunc)
    {
        algo::transform(inArray.begin(), inArray.end(), inArray.begin(), inFunc);
    }
}
