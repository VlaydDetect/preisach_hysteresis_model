// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/Constants.hpp"
#include "Functions/linspace.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// The Blackman window is a taper formed by using the first three terms of a summation of
    /// cosines. It was designed to have close to the minimal leakage possible. It is close to
    /// optimal, only slightly worse than a Kaiser window.
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @return Matrix
    inline Matrix<double> blackman(int32 m)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble = static_cast<double>(m);

        Matrix<double> result(1, m);
        int32 i = 0;
        for (auto n : linspace(0., mDouble, m, EndPoint::YES))
        {
            const auto nOverM = n / mDouble;
            result[i++] =
                0.42 - 0.5 * std::cos(2. * consts::pi * nOverM) + 0.08 * std::cos(4. * consts::pi * nOverM);
        }

        return result;
    }
}
