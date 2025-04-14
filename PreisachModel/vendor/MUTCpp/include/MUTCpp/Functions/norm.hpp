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
    /// Matrix or vector norm.
    template <typename dtype>
    Matrix<double> norm(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double sumOfSquares = 0.;
        const auto function = [&sumOfSquares](dtype value) -> void
        {
            sumOfSquares += utils::sqr(static_cast<double>(value));
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                Matrix returnArray = {std::sqrt(sumOfSquares)};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<double> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    sumOfSquares = 0.;
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(sumOfSquares);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return norm(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
    
    /// Matrix or vector norm.
    template <typename dtype>
    Matrix<std::complex<double>> norm(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> sumOfSquares(0., 0.);
        const auto function = [&sumOfSquares](const std::complex<dtype> &value) -> void
        {
            sumOfSquares += utils::sqr(complex_cast<double>(value));
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                Matrix<std::complex<double>> returnArray = {std::sqrt(sumOfSquares)};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<std::complex<double>> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    sumOfSquares = std::complex<double>(0., 0.);
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(sumOfSquares);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return norm(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
}
