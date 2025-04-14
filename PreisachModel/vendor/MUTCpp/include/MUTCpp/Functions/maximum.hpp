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
#include "Matrix/MatrixBroadcast.hpp"

namespace mc
{
    /// Element-wise maximum of array elements.
    template <typename dtype>
    Matrix<dtype> maximum(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto comparitor = [](const dtype &lhs, const dtype &rhs) noexcept -> bool { return lhs < rhs; };
        return broadcast::broadcaster<dtype>(inArray1,
                                             inArray2,
                                             [&comparitor](const dtype &value1, const dtype &value2)
                                             {
                                                 return std::max(value1, value2, comparitor);
                                             });
    }

    /// Element-wise maximum of array elements.
    template <typename dtype>
    Matrix<dtype> maximum(const Matrix<dtype> &inArray, const dtype &inScalar)
    {
        const Matrix<dtype> inArray2 = {inScalar};
        return maximum(inArray, inArray2);
    }

    /// Element-wise maximum of array elements.
    template <typename dtype>
    Matrix<dtype> maximum(const dtype &inScalar, const Matrix<dtype> &inArray)
    {
        return maximum(inArray, inScalar);
    }
}
