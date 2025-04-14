// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// The outer product of two vectors.  Inputs are flattened if not already 1-dimensional.
    template <typename dtype>
    Matrix<dtype> outer(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto size = inArray1.size();

        if (size != inArray2.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input arrays must be the same length");
        }

        auto returnArray = Matrix<dtype>(size);
        for (uint32 row = 0; row < size; ++row)
        {
            const auto array1Value = inArray1[row];

            std::transform(inArray2.begin(),
                           inArray2.end(),
                           returnArray.begin(row),
                           [array1Value](dtype value) -> dtype { return array1Value * value; });
        }

        return returnArray;
    }
}
