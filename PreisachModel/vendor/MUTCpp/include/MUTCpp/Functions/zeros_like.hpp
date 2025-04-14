// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a new matrix of given shape and type, filled with zeros.
    /// @return Matrix
    template<typename dtypeOut, typename dtype>
    Matrix<dtypeOut> zeros_like(const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<dtypeOut> returnArray(inArray.shape());
        returnArray.zeros();
        return returnArray;
    }
}
