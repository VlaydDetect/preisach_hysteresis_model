// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns a complex number with magnitude r and phase angle theta.
    template <typename dtype>
    auto polar(dtype magnitude, dtype phaseAngle)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::polar(magnitude, phaseAngle);
    }

    /// Returns a complex number with magnitude r and phase angle theta.
    template <typename dtype>
    auto polar(const Matrix<dtype> &magnitude, const Matrix<dtype> &phaseAngle)
    {
        if (magnitude.shape() != phaseAngle.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input magnitude and phaseAngle arrays must be the same shape");
        }

        Matrix<decltype(mc::polar(dtype{0}, dtype{0}))> returnArray(magnitude.shape());
        algo::transform(magnitude.cbegin(),
                        magnitude.cend(),
                        phaseAngle.begin(),
                        returnArray.begin(),
                        [](dtype mag, dtype angle) -> auto { return mc::polar(mag, angle); });

        return returnArray;
    }
}
