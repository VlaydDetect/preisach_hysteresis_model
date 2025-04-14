// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <initializer_list>
#include <string>
#include <vector>

#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Functions/column_stack.hpp"
#include "Functions/row_stack.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace detail
    {
        /// Compute the variance along the specified axis.
        /// @param begin: iterator to the beginning of the span
        /// @param end: iterator to one past the end of the span
        /// @param inAxis: the axis to stack
        /// @return Matrix
        template <typename dtype, typename Iterator>
        Matrix<dtype> stack(Iterator begin, Iterator end, Axis inAxis)
        {
            switch (inAxis)
            {
            case Axis::ROW:
            {
                return row_stack<dtype>(begin, end);
            }
            case Axis::COL:
            {
                return column_stack<dtype>(begin, end);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("inAxis must be either ROW or COL.");
                return {}; // getting rid of compiler warning
            }
            }
        }
    }

    /// Compute the variance along the specified axis.
    /// @param inArrayList: {list} of arrays to stack
    /// @param inAxis: axis to stack the input Matrixs
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> stack(std::initializer_list<Matrix<dtype>> inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::stack<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }

    /// Compute the variance along the specified axis.
    /// @param inArrayList: {list} of arrays to stack
    /// @param inAxis: axis to stack the input Matrixs
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> stack(std::vector<Matrix<dtype>> inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::stack<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }
}
