// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Enums.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Functions/linspace.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Return numbers spaced evenly on a log scale.
    ///
    /// This is similar to logspace, but with endpoints specified directly.
    /// Each output sample is a constant multiple of the previous.
    ///
    /// @param start: the starting value of a sequence
    /// @param stop: The final value of the sequence, unless endpoint is False.
    /// In that case, num + 1 values are spaced over the interval
    /// in log-space, of which all but the last (a sequence of length num) are returned.
    /// @param num: Number of samples to generate. Default 50.
    /// @param endPoint: If true, stop is the last sample. Otherwise, it is not included. Default is true.
    /// @param base: The base of the log space. The step size between the elements in ln(samples) / ln(base)
    /// @return Matrix
    template <typename dtype>
    Matrix<double>
    logspace(dtype start, dtype stop, uint32 num = 50, EndPoint endPoint = EndPoint::YES, double base = 10.)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        auto spacedValues = linspace(static_cast<double>(start), static_cast<double>(stop), num, endPoint);
        algo::for_each(spacedValues.begin(),
                       spacedValues.end(),
                       [base](auto &value) -> void { value = utils::powerf(base, value); });

        return spacedValues;
    }
}
