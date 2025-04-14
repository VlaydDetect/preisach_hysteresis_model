// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include "Core/Types.hpp"
#include "Functions/mean.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Compute the standard deviation along the specified axis.
    template <typename dtype>
    Matrix<double> stdev(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double meanValue = 0.;
        double sum = 0.;

        const auto function = [&sum, &meanValue](dtype value) -> void
        {
            sum += utils::sqr(static_cast<double>(value) - meanValue);
        };

        switch (inAxis)
        {
        case Axis::NONE:
        {
            meanValue = mean(inArray, inAxis).item();
            std::for_each(inArray.cbegin(), inArray.cend(), function);

            Matrix<double> returnArray = {std::sqrt(sum / inArray.size())};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> meanValueArray = mean(inArray, inAxis);
            Matrix<double> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                meanValue = meanValueArray[row];
                sum = 0.;
                std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                returnArray(0, row) = std::sqrt(sum / inArray.numCols());
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return stdev(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }

    /// Compute the standard deviation along the specified axis.
    template <typename dtype>
    Matrix<std::complex<double>> stdev(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> meanValue(0., 0.);
        std::complex<double> sum(0., 0.);

        const auto function = [&sum, &meanValue](std::complex<dtype> value) -> void
        {
            sum += utils::sqr(complex_cast<double>(value) - meanValue);
        };

        switch (inAxis)
        {
        case Axis::NONE:
        {
            meanValue = mean(inArray, inAxis).item();
            std::for_each(inArray.cbegin(), inArray.cend(), function);

            Matrix<std::complex<double>> returnArray = {std::sqrt(sum / static_cast<double>(inArray.size()))};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<std::complex<double>> meanValueArray = mean(inArray, inAxis);
            Matrix<std::complex<double>> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                meanValue = meanValueArray[row];
                sum = std::complex<double>(0., 0.);
                std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                returnArray(0, row) = std::sqrt(sum / static_cast<double>(inArray.numCols()));
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            Matrix<std::complex<double>> meanValueArray = mean(inArray, inAxis);
            Matrix<std::complex<dtype>> transposedArray = inArray.transpose();
            Matrix<std::complex<double>> returnArray(1, transposedArray.numRows());
            for (uint32 row = 0; row < transposedArray.numRows(); ++row)
            {
                meanValue = meanValueArray[row];
                sum = std::complex<double>(0., 0.);
                std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);

                returnArray(0, row) = std::sqrt(sum / static_cast<double>(transposedArray.numCols()));
            }

            return returnArray;
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
