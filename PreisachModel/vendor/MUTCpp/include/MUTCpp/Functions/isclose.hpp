// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns a boolean array where two arrays are element-wise
    /// equal within a tolerance.
    /// For finite values, isclose uses the following equation to test whether two floating point values are equivalent.
    /// absolute(a - b) <= (atol + rtol * absolute(b))
    /// @param inArray1
    /// @param inArray2
    /// @param inRtol: relative tolerance (default 1e-5)
    /// @param inAtol: absolute tolerance (default 1e-9)
    template <typename dtype>
    Matrix<bool> isclose(const Matrix<dtype> &inArray1,
                         const Matrix<dtype> &inArray2,
                         double inRtol = 1e-05,
                         double inAtol = 1e-08)
    {
        STATIC_ASSERT_FLOAT(dtype);

        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        Matrix<bool> returnArray(inArray1.shape());
        algo::transform(inArray1.cbegin(),
                        inArray1.cend(),
                        inArray2.cbegin(),
                        returnArray.begin(),
                        [inRtol, inAtol](dtype inValueA, dtype inValueB) noexcept -> bool
                        {
                            return std::abs(inValueA - inValueB) <= (inAtol + inRtol * std::abs(inValueB));
                        });

        return returnArray;
    }
}
