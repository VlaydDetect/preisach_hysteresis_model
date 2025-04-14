// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <iostream>

#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Prints the matrix to the console.
    template<typename dtype>
    void print(const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        std::cout << inArray;
    }
}
