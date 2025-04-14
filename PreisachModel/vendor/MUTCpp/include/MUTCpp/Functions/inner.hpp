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
    /// Inner product of two 1-D arrays.
    template<typename dtype>
    dtype inner(const Matrix<dtype>& a, const Matrix<dtype>& b)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (a.size() != b.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Inputs 'a' and 'b' must have the same size");
        }

        return std::inner_product(a.cbegin(), a.cend(), b.cbegin(), dtype{ 0 });
    }
}
