// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Roll array elements along a given axis.
    /// @param inArray
    /// @param inShift: (elements to shift, positive means forward, negative means backwards)
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> roll(const Matrix<dtype> &inArray, int32 inShift, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
        case Axis::NONE:
        {
            uint32 shift = std::abs(inShift) % inArray.size();
            if (inShift > 0)
            {
                shift = inArray.size() - shift;
            }

            Matrix<dtype> returnArray(inArray);
            algo::rotate(returnArray.begin(), returnArray.begin() + shift, returnArray.end());

            return returnArray;
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();

            uint32 shift = std::abs(inShift) % inShape.m_Cols;
            if (inShift > 0)
            {
                shift = inShape.m_Cols - shift;
            }

            Matrix<dtype> returnArray(inArray);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                algo::rotate(returnArray.begin(row),
                             returnArray.begin(row) + shift,
                             returnArray.end(row));
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return roll(inArray.transpose(), inShift, Axis::COL).transpose();
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
