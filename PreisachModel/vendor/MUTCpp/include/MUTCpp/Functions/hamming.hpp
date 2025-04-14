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
    /// Return the Hamming window.
    /// The Hamming window is a taper formed by using a weighted cosine.
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @return Matrix
    inline Matrix<double> hamming(int32 m)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble = static_cast<double>(m);
        const auto twoPiDivMMinus1 = (2. * consts::pi) / (mDouble - 1.);

        Matrix<double> result(1, m);
        int32 i = 0;
        for (auto n : linspace(0., mDouble - 1., m, EndPoint::YES))
        {
            result[i++] = 0.54 - 0.46 * std::cos(twoPiDivMMinus1 * n);
        }

        return result;
    }
}
