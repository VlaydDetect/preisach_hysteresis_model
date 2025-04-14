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
    /// Return the angle of the complex argument.
    template <typename dtype>
    auto angle(const std::complex<dtype> &inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::arg(inValue);
    }

    /// Return the angle of the complex argument.
    template <typename dtype>
    auto angle(const Matrix<std::complex<dtype>> &inArray)
    {
        Matrix<decltype(angle(std::complex<dtype>{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](auto &inValue) -> auto { return angle(inValue); });

        return returnArray;
    }
}
