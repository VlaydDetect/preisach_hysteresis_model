// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// maximum of inputs.
    ///
    /// Compare two values and return a value containing the
    /// maxima
    template<typename dtype>
    dtype fmax(dtype inValue1, dtype inValue2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::max(inValue1,
                        inValue2,
                        [](const dtype value1, const dtype value2) noexcept -> bool { return value1 < value2; });
    }
    
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and return a new array containing the
    /// element - wise maxima
    template<typename dtype>
    Matrix<dtype> fmax(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        return broadcast::broadcaster<dtype>(inArray1,
                                             inArray2,
                                             [](dtype inValue1, dtype inValue2) noexcept -> dtype
                                             { return fmax(inValue1, inValue2); });
    }
    
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and return a new array containing the
    /// element - wise maxima
    template<typename dtype>
    Matrix<dtype> fmax(const Matrix<dtype>& inArray, const dtype& inScalar)
    {
        const Matrix<dtype> inArray2 = { inScalar };
        return fmax(inArray, inArray2);
    }
    
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and return a new array containing the
    /// element - wise maxima
    template<typename dtype>
    Matrix<dtype> fmax(const dtype& inScalar, const Matrix<dtype>& inArray)
    {
        return fmax(inArray, inScalar);
    }
}
