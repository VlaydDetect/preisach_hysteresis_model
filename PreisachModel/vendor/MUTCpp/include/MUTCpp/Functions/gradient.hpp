// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/ComplexOperators.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the gradient of the array.
    template <typename dtype>
    Matrix<double> gradient(const Matrix<dtype> &inArray, Axis inAxis = Axis::ROW)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
        case Axis::ROW:
        {
            const auto inShape = inArray.shape();
            if (inShape.rows < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 row.");
            }

            // first do the first and last rows
            auto returnArray = Matrix<double>(inShape);
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                returnArray(0, col) = static_cast<double>(inArray(1, col)) - static_cast<double>(inArray(0, col));
                returnArray(-1, col) =
                    static_cast<double>(inArray(-1, col)) - static_cast<double>(inArray(-2, col));
            }

            // then rip through the rest of the array
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                for (uint32 row = 1; row < inShape.rows - 1; ++row)
                {
                    returnArray(row, col) =
                        (static_cast<double>(inArray(row + 1, col)) - static_cast<double>(inArray(row - 1, col))) /
                        2.;
                }
            }

            return returnArray;
        }
        case Axis::COL:
        {
            const auto inShape = inArray.shape();
            if (inShape.cols < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 columns.");
            }

            // first do the first and last columns
            auto returnArray = Matrix<double>(inShape);
            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                returnArray(row, 0) = static_cast<double>(inArray(row, 1)) - static_cast<double>(inArray(row, 0));
                returnArray(row, -1) =
                    static_cast<double>(inArray(row, -1)) - static_cast<double>(inArray(row, -2));
            }

            // then rip through the rest of the array
            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                for (uint32 col = 1; col < inShape.cols - 1; ++col)
                {
                    returnArray(row, col) =
                        (static_cast<double>(inArray(row, col + 1)) - static_cast<double>(inArray(row, col - 1))) /
                        2.;
                }
            }

            return returnArray;
        }
        default:
        {
            // will return the gradient of the flattened array
            if (inArray.size() < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 element.");
            }

            auto returnArray = Matrix<double>(1, inArray.size());
            returnArray[0] = static_cast<double>(inArray[1]) - static_cast<double>(inArray[0]);
            returnArray[-1] = static_cast<double>(inArray[-1]) - static_cast<double>(inArray[-2]);

            algo::transform(inArray.cbegin() + 2,
                            inArray.cend(),
                            inArray.cbegin(),
                            returnArray.begin() + 1,
                            [](dtype value1, dtype value2) -> double
                            {
                                return (static_cast<double>(value1) - static_cast<double>(value2)) / 2.;
                            });

            return returnArray;
        }
        }
    }

    /// Return the gradient of the array.
    template <typename dtype>
    Matrix<std::complex<double>> gradient(const Matrix<std::complex<dtype>> &inArray, Axis inAxis = Axis::ROW)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
        case Axis::ROW:
        {
            const auto inShape = inArray.shape();
            if (inShape.rows < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 row.");
            }

            // first do the first and last rows
            auto returnArray = Matrix<std::complex<double>>(inShape);
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                returnArray(0, col) = complex_cast<double>(inArray(1, col)) - complex_cast<double>(inArray(0, col));
                returnArray(-1, col) =
                    complex_cast<double>(inArray(-1, col)) - complex_cast<double>(inArray(-2, col));
            }

            // then rip through the rest of the array
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                for (uint32 row = 1; row < inShape.rows - 1; ++row)
                {
                    returnArray(row, col) = (complex_cast<double>(inArray(row + 1, col)) -
                            complex_cast<double>(inArray(row - 1, col))) /
                        2.;
                }
            }

            return returnArray;
        }
        case Axis::COL:
        {
            const auto inShape = inArray.shape();
            if (inShape.cols < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 columns.");
            }

            // first do the first and last columns
            auto returnArray = Matrix<std::complex<double>>(inShape);
            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                returnArray(row, 0) = complex_cast<double>(inArray(row, 1)) - complex_cast<double>(inArray(row, 0));
                returnArray(row, -1) =
                    complex_cast<double>(inArray(row, -1)) - complex_cast<double>(inArray(row, -2));
            }

            // then rip through the rest of the array
            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                for (uint32 col = 1; col < inShape.cols - 1; ++col)
                {
                    returnArray(row, col) = (complex_cast<double>(inArray(row, col + 1)) -
                            complex_cast<double>(inArray(row, col - 1))) /
                        2.;
                }
            }

            return returnArray;
        }
        default:
        {
            // will return the gradient of the flattened array
            if (inArray.size() < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 element.");
            }

            auto returnArray = Matrix<std::complex<double>>(1, inArray.size());
            returnArray[0] = complex_cast<double>(inArray[1]) - complex_cast<double>(inArray[0]);
            returnArray[-1] = complex_cast<double>(inArray[-1]) - complex_cast<double>(inArray[-2]);

            algo::transform(
                inArray.cbegin() + 2,
                inArray.cend(),
                inArray.cbegin(),
                returnArray.begin() + 1,
                [](const std::complex<dtype> &value1, const std::complex<dtype> &value2) -> std::complex<double>
                {
                    return (complex_cast<double>(value1) - complex_cast<double>(value2)) / 2.;
                });

            return returnArray;
        }

        }
    }
}
