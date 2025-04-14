// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Functions/linspace.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// The Bartlett window is very similar to a triangular window, except that the end
    /// points are at zero. It is often used in signal processing for tapering a signal,
    /// without generating too much ripple in the frequency domain.
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @return Matrix
    inline Matrix<double> bartlett(int32 m)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble = static_cast<double>(m);
        const auto mMinus1Over2 = (mDouble - 1.) / 2.;
        const auto mMinus1Over2Inv = 1. / mMinus1Over2;

        Matrix<double> result(1, m);
        int32 i = 0;
        for (auto n : linspace(0., mDouble - 1., m, EndPoint::YES))
        {
            result[i++] = mMinus1Over2Inv * (mMinus1Over2 - std::abs(n - mMinus1Over2));
        }

        return result;
    }
}
