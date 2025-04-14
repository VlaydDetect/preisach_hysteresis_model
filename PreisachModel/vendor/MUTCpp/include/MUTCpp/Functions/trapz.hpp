// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Integrate along the given axis using the composite trapezoidal rule.
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    /// @param inArray
    /// @param dx: (Optional defaults to 1.)
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    template <typename dtype>
    Matrix<double> trapz(const Matrix<dtype> &inArray, double dx = 1., Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const Shape inShape = inArray.shape();
        switch (inAxis)
        {
        case Axis::NONE:
        {
            double sum = 0.;
            for (uint32 i = 0; i < inArray.size() - 1; ++i)
            {
                sum += static_cast<double>(inArray[i + 1] - inArray[i]) / 2. + static_cast<double>(inArray[i]);
            }

            Matrix<double> returnArray = {sum * dx};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> returnArray(inShape.m_Rows, 1);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                double sum = 0;
                for (uint32 col = 0; col < inShape.m_Cols - 1; ++col)
                {
                    sum += static_cast<double>(inArray(row, col + 1) - inArray(row, col)) / 2. +
                        static_cast<double>(inArray(row, col));
                }

                returnArray[row] = sum * dx;
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return trapz(inArray.transpose(), dx, Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }

    /// Integrate along the given axis using the composite trapezoidal rule.
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    template <typename dtype>
    Matrix<double> trapz(const Matrix<dtype> &inArrayY, const Matrix<dtype> &inArrayX, Axis inAxis = Axis::NONE)
    {
        const Shape inShapeY = inArrayY.shape();
        const Shape inShapeX = inArrayX.shape();

        if (inShapeY != inShapeX)
        {
            THROW_INVALID_ARGUMENT_ERROR("input x and y arrays should be the same shape.");
        }

        switch (inAxis)
        {
        case Axis::NONE:
        {
            double sum = 0.;
            for (uint32 i = 0; i < inArrayY.size() - 1; ++i)
            {
                const auto dx = static_cast<double>(inArrayX[i + 1] - inArrayX[i]);
                sum += dx *
                    (static_cast<double>(inArrayY[i + 1] - inArrayY[i]) / 2. + static_cast<double>(inArrayY[i]));
            }

            Matrix<double> returnArray = {sum};
            return returnArray;
        }
        case Axis::COL:
        {
            Matrix<double> returnArray(inShapeY.m_Rows, 1);
            for (uint32 row = 0; row < inShapeY.m_Rows; ++row)
            {
                double sum = 0;
                for (uint32 col = 0; col < inShapeY.m_Cols - 1; ++col)
                {
                    const auto dx = static_cast<double>(inArrayX(row, col + 1) - inArrayX(row, col));
                    sum += dx * (static_cast<double>(inArrayY(row, col + 1) - inArrayY(row, col)) / 2. +
                        static_cast<double>(inArrayY(row, col)));
                }

                returnArray[row] = sum;
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return trapz(inArrayY.transpose(), inArrayX.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
