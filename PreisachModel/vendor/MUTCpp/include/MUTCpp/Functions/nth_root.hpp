// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Return the nth-root of a value.
    template <typename dtype1, typename dtype2>
    double nth_root(dtype1 inValue, dtype2 inRoot) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype1);
        STATIC_ASSERT_ARITHMETIC(dtype2);

        return utils::powerf(static_cast<double>(inValue), 1. / static_cast<double>(inRoot));
    }

    /// Return the nth-root of an array.
    template <typename dtype1, typename dtype2>
    Matrix<double> nth_root(const Matrix<dtype1> &inArray, dtype2 inRoot)
    {
        Matrix<double> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [inRoot](dtype1 inValue) noexcept -> double { return nth_root(inValue, inRoot); });

        return returnArray;
    }
}
