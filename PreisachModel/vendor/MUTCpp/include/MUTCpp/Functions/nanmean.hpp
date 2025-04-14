// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the mean along the specified axis ignoring NaNs.
    template <typename dtype>
    Matrix<double> nanmean(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            auto sum = static_cast<double>(std::accumulate(inArray.cbegin(),
                                                           inArray.cend(),
                                                           0.,
                                                           [](dtype inValue1, dtype inValue2) -> dtype
                                                           {
                                                               return std::isnan(inValue2)
                                                                   ? inValue1
                                                                   : inValue1 + inValue2;
                                                           }));

            const auto numberNonNan =
                static_cast<double>(std::accumulate(inArray.cbegin(),
                                                    inArray.cend(),
                                                    0.,
                                                    [](dtype inValue1, dtype inValue2) -> dtype
                                                    {
                                                        return std::isnan(inValue2) ? inValue1 : inValue1 + 1;
                                                    }));

            Matrix<double> returnArray = {sum /= numberNonNan};

            return returnArray;
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();
            Matrix<double> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                auto sum = static_cast<double>(
                    std::accumulate(inArray.cbegin(row),
                                    inArray.cend(row),
                                    0.,
                                    [](dtype inValue1, dtype inValue2) -> dtype
                                    {
                                        return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2;
                                    }));

                auto numberNonNan =
                    static_cast<double>(std::accumulate(inArray.cbegin(row),
                                                        inArray.cend(row),
                                                        0.,
                                                        [](dtype inValue1, dtype inValue2) -> dtype
                                                        {
                                                            return std::isnan(inValue2) ? inValue1 : inValue1 + 1;
                                                        }));

                returnArray(0, row) = sum / numberNonNan;
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return nanmean(inArray.transpose(), Axis::COL);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
