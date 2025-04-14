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
    /// Return the real part of the complex argument.
    template <typename dtype>
    auto real(const std::complex<dtype> &inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::real(inValue);
    }

    /// Return the real part of the complex argument.
    template <typename dtype>
    auto real(const Matrix<std::complex<dtype>> &inArray)
    {
        Matrix<decltype(mc::real(std::complex<dtype>{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](auto &inValue) -> auto { return mc::real(inValue); });

        return returnArray;
    }
}
