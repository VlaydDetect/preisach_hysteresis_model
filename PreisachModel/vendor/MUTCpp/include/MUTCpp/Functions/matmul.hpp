// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Functions/dot.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Matrix product of two arrays.
    template<typename dtype>
    Matrix<dtype> matmul(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        return dot(inArray1, inArray2);
    }

    /// Matrix product of two arrays.
    template<typename dtype>
    Matrix<std::complex<dtype>> matmul(const Matrix<dtype>& inArray1, const Matrix<std::complex<dtype>>& inArray2)
    {
        return dot(inArray1, inArray2);
    }

    /// Matrix product of two arrays.
    template<typename dtype>
    Matrix<std::complex<dtype>> matmul(const Matrix<std::complex<dtype>>& inArray1, const Matrix<dtype>& inArray2)
    {
        return dot(inArray1, inArray2);
    }
}
