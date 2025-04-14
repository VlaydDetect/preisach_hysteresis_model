// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>
#include <numeric>

#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the mean along the specified axis.
    template <typename dtype>
    Matrix<double> mean(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            auto sum = std::accumulate(inArray.cbegin(), inArray.cend(), 0.);
            Matrix<double> returnArray = {sum /= static_cast<double>(inArray.size())};

            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                auto sum = std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.);
                returnArray(0, row) = sum / static_cast<double>(inArray.numCols());
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return mean(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {};
        }
        }
    }

    //============================================================================
    // Method Description:
    /// Compute the mean along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mean.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return Matrix
    ///
    template <typename dtype>
    Matrix<std::complex<double>> mean(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            auto sum = std::accumulate(inArray.cbegin(), inArray.cend(), std::complex<double>(0.));
            Matrix<std::complex<double>> returnArray = {sum /= std::complex<double>(inArray.size())};

            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<std::complex<double>> returnArray(1, inArray.numRows());
            for (uint32 row = 0; row < inArray.numRows(); ++row)
            {
                auto sum = std::accumulate(inArray.cbegin(row), inArray.cend(row), std::complex<double>(0.));
                returnArray(0, row) = sum / std::complex<double>(inArray.numCols());
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            Matrix<std::complex<double>> transposedArray = inArray.transpose();
            Matrix<std::complex<double>> returnArray(1, transposedArray.numRows());
            for (uint32 row = 0; row < transposedArray.numRows(); ++row)
            {
                auto sum = std::accumulate(transposedArray.cbegin(row),
                                           transposedArray.cend(row),
                                           std::complex<double>(0.));
                returnArray(0, row) = sum / std::complex<double>(transposedArray.numCols());
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
