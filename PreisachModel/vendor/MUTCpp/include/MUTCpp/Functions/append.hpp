// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Debug/Profile.hpp"

namespace mc
{
    /// Append values to the end of an array.
    /// @param inArray
    /// @param inAppendValues
    /// @param inAxis (Optional, default NONE): The axis along which values are appended.
    /// If axis is not given, both inArray and inAppendValues
    /// are flattened before use.
    /// @return Matrix
    template<typename dtype>
    Matrix<dtype> append(Matrix<dtype>& inArray, const Matrix<dtype>& inAppendValues, Axis inAxis = Axis::NONE)
    {
        AL_PROFILE_FUNC("Matrix::append");
        if (inArray.shape().isnull())
        {
            return std::move(inAppendValues);
        }
        if (inAppendValues.shape().isnull())
        {
            return std::move(inArray);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                Matrix<dtype> returnArray(1, inArray.size() + inAppendValues.size());
                algo::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                algo::copy(inAppendValues.cbegin(),
                                     inAppendValues.cend(),
                                     returnArray.begin() + inArray.size());
                
                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape     = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.m_Cols != appendShape.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                Matrix<dtype> returnArray(inShape.m_Rows + appendShape.m_Rows, inShape.m_Cols);
                algo::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                algo::copy(inAppendValues.cbegin(),
                                     inAppendValues.cend(),
                                     returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape     = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.m_Rows != appendShape.m_Rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                Matrix<dtype> returnArray(inShape.m_Rows, inShape.m_Cols + appendShape.m_Cols);
                for (uint32 row = 0; row < returnArray.shape().m_Rows; ++row)
                {
                    algo::copy(inArray.cbegin(row), inArray.cend(row), returnArray.begin(row));
                    algo::copy(inAppendValues.cbegin(row),
                                         inAppendValues.cend(row),
                                         returnArray.begin(row) + inShape.m_Cols);
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
