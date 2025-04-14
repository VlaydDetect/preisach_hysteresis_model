// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <complex>

#include "Core/StaticAssert.hpp"
#include "Functions/stdev.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the variance along the specified axis.
    template <typename dtype>
    Matrix<double> var(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<double> stdValues = stdev(inArray, inAxis);
        const auto function = [](double &value) -> void { value *= value; };

        algo::for_each(stdValues.begin(), stdValues.end(), function);
        return stdValues;
    }

    /// Compute the variance along the specified axis.
    template <typename dtype>
    Matrix<std::complex<double>> var(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<std::complex<double>> stdValues = stdev(inArray, inAxis);
        const auto function = [](std::complex<double> &value) -> void { value *= value; };

        algo::for_each(stdValues.begin(), stdValues.end(), function);
        return stdValues;
    }
}
