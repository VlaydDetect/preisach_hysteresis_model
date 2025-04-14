// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Compute the root-mean-square (RMS) along the specified axis.
    template <typename dtype>
    Matrix<double> rms(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double squareSum = 0.;
        const auto function = [&squareSum](dtype value) -> void
        {
            squareSum += utils::sqr(static_cast<double>(value));
        };

        switch (inAxis)
        {
        case Axis::NONE:
        {
            std::for_each(inArray.cbegin(), inArray.cend(), function);
            Matrix<double> returnArray = {std::sqrt(squareSum / static_cast<double>(inArray.size()))};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                squareSum = 0.;
                std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(inArray.numCols()));
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return rms(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }

    /// Compute the root-mean-square (RMS) along the specified axis.
    template <typename dtype>
    Matrix<std::complex<double>> rms(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> squareSum = 0.;
        const auto function = [&squareSum](std::complex<dtype> value) -> void
        {
            squareSum += utils::sqr(complex_cast<double>(value));
        };

        switch (inAxis)
        {
        case Axis::NONE:
        {
            std::for_each(inArray.cbegin(), inArray.cend(), function);
            Matrix<std::complex<double>> returnArray = {std::sqrt(squareSum /
                static_cast<double>(inArray.size()))};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<std::complex<double>> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                squareSum = std::complex<double>(0., 0.);
                std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(inArray.numCols()));
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return rms(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
