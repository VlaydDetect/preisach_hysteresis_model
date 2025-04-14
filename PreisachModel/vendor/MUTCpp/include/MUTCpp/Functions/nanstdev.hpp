// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>

#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/nanmean.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Compute the standard deviation along the specified axis, while ignoring NaNs.
    template <typename dtype>
    Matrix<double> nanstdev(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            double meanValue = nanmean(inArray, inAxis).item();
            double sum = 0;
            double counter = 0;
            for (auto value : inArray)
            {
                if (std::isnan(value))
                {
                    continue;
                }

                sum += utils::sqr(static_cast<double>(value) - meanValue);
                ++counter;
            }
            Matrix<double> returnArray = {std::sqrt(sum / counter)};
            return returnArray;
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();
            Matrix<double> meanValue = nanmean(inArray, inAxis);
            Matrix<double> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                double sum = 0;
                double counter = 0;
                for (uint32 col = 0; col < inShape.m_Cols; ++col)
                {
                    if (std::isnan(inArray(row, col)))
                    {
                        continue;
                    }

                    sum += utils::sqr(static_cast<double>(inArray(row, col)) - meanValue[row]);
                    ++counter;
                }
                returnArray(0, row) = std::sqrt(sum / counter);
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return nanstdev(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
