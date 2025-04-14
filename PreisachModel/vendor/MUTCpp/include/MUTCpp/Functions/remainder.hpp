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
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return a remainder of division.
    template <typename dtype>
    double remainder(dtype inValue1, dtype inValue2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return static_cast<double>(std::remainder(inValue1, inValue2));
    }

    /// Return element-wise remainder of the division.
    template <typename dtype>
    Matrix<double> remainder(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        return broadcast::broadcaster<double>(inArray1,
                                              inArray2,
                                              [](dtype inValue1, dtype inValue2) noexcept -> double
                                              {
                                                  return remainder(inValue1, inValue2);
                                              });
    }
}
