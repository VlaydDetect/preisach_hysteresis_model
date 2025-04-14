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
    /// Returns the projection of the complex number z onto the Riemann sphere.
    template <typename dtype>
    auto proj(const std::complex<dtype> &inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::proj(inValue);
    }

    /// Returns the projection of the complex number z onto the Riemann sphere.
    template <typename dtype>
    auto proj(const Matrix<std::complex<dtype>> &inArray)
    {
        Matrix<decltype(mc::proj(std::complex<dtype>{0}))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](auto &inValue) -> auto { return mc::proj(inValue); });

        return returnArray;
    }
}
