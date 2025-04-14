// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <vector>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the median along the specified axis ignoring NaNs.
    template <typename dtype>
    Matrix<dtype> nanmedian(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            std::vector<dtype> values;
            for (auto value : inArray)
            {
                if (!std::isnan(value))
                {
                    values.push_back(value);
                }
            }

            const uint32 middle = static_cast<uint32>(values.size()) / 2;
            algo::nth_element(values.begin(), values.begin() + middle, values.end());
            Matrix<dtype> returnArray = {values[middle]};

            return returnArray;
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();
            Matrix<dtype> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                std::vector<dtype> values;
                for (uint32 col = 0; col < inShape.m_Cols; ++col)
                {
                    if (!std::isnan(inArray(row, col)))
                    {
                        values.push_back(inArray(row, col));
                    }
                }

                const uint32 middle = static_cast<uint32>(values.size()) / 2;
                algo::nth_element(values.begin(), values.begin() + middle, values.end());
                returnArray(0, row) = values[middle];
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return nanmedian(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
