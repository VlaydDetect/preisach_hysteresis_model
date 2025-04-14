// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#if defined(__cpp_lib_math_special_functions)

#include "Functions/linspace.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    namespace detail
    {
        /// Modified Cylindrical Bessel function of the first kind.
        /// @param inV: the order of the bessel function
        /// @param inX: the input value
        /// @return calculated-result-type
        template <typename dtype1, typename dtype2>
        auto bessel_in(dtype1 inV, dtype2 inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return std::cyl_bessel_i(static_cast<double>(inV), static_cast<double>(inX));
        }
    }

    /// The Kaiser window is a taper formed by using a Bessel function.
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @param beta: shape parameter for the window
    /// @return Matrix
    inline Matrix<double> kaiser(int32 m, double beta)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble = static_cast<double>(m);
        const auto mMinus1 = mDouble - 1.;
        const auto mMinus1Over2 = mMinus1 / 2.;
        const auto mMinus1Squared = utils::sqr(mMinus1);
        const auto i0Beta = detail::bessel_in(0, beta);

        Matrix<double> result(1, m);
        int32 i = 0;
        for (auto n : linspace(-mMinus1Over2, mMinus1Over2, m, EndPoint::YES))
        {
            auto value = beta * std::sqrt(1. - (4. * utils::sqr(n)) / mMinus1Squared);
            result[i++] = detail::bessel_in(0, value) / i0Beta;
        }

        return result;
    }
}

#endif
