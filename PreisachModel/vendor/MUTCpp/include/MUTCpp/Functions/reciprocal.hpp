// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/ComplexOperators.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the reciprocal of the argument, element-wise.
    ///
    /// Calculates 1 / x.
    template <typename dtype>
    Matrix<double> reciprocal(const Matrix<dtype> &inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<double> returnArray(inArray.shape());

        uint32 counter = 0;
        std::for_each(inArray.cbegin(),
                      inArray.cend(),
                      [&returnArray, &counter](dtype value) noexcept -> void
                      {
                          returnArray[counter++] = 1. / static_cast<double>(value);
                      });

        return returnArray;
    }

    /// Return the reciprocal of the argument, element-wise.
    ///
    /// Calculates 1 / x.
    template <typename dtype>
    Matrix<std::complex<double>> reciprocal(const Matrix<std::complex<dtype>> &inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<std::complex<double>> returnArray(inArray.shape());

        uint32 counter = 0;
        std::for_each(inArray.cbegin(),
                      inArray.cend(),
                      [&returnArray, &counter](std::complex<dtype> value) -> void
                      {
                          returnArray[counter++] = std::complex<double>(1.) / complex_cast<double>(value);
                      });

        return returnArray;
    }
}
