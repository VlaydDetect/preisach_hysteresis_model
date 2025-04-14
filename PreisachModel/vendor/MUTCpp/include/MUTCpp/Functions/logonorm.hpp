// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"
#include "log.hpp"
#include "norm.hpp"

namespace mc
{
    /// Natural logarithm of matrix or vector norm
    template <typename dtype>
    Matrix<double> logonorm(const Matrix<dtype> &inArray, Axis inNormAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);
        
        return mc::log(mc::norm(inArray, inNormAxis));
    }

    /// Natural logarithm of matrix or vector norm
    template <typename dtype>
    Matrix<std::complex<double>> logonorm(const Matrix<std::complex<dtype>> &inArray, Axis inNormAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);
        
        return mc::log(mc::norm(inArray, inNormAxis));
    }
}
