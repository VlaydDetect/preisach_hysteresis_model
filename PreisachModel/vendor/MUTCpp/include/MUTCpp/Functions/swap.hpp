// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Swaps the contents of two arrays
    template <typename dtype>
    void swap(Matrix<dtype> &inArray1, Matrix<dtype> &inArray2) noexcept
    {
        Matrix<dtype> tmp(std::move(inArray1));
        inArray1 = std::move(inArray2);
        inArray2 = std::move(tmp);
    }
}
