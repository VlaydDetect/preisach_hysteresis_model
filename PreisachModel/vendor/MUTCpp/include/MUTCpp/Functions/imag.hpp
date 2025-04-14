// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the imaginar  part of the complex argument.
    template <typename dtype>
    auto imag(const std::complex<dtype> &inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::imag(inValue);
    }

    /// Return the imaginary part of the complex argument.
    template <typename dtype>
    auto imag(const Matrix<std::complex<dtype>> &inArray)
    {
        Matrix<decltype(mc::imag(std::complex<dtype>{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](auto &inValue) -> auto { return mc::imag(inValue); });

        return returnArray;
    }
}
