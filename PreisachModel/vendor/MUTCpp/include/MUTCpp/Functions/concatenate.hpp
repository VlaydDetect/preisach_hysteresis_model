// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <initializer_list>
#include <vector>

#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/column_stack.hpp"
#include "Functions/row_stack.hpp"

namespace mc
{
    namespace detail
    {
        /// Join a sequence of arrays along an existing axis.
        /// @param begin: the begin iterator
        /// @param end: the end iterator
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        template <typename dtype, typename Iterator>
        Matrix<dtype> concatenate(Iterator begin, Iterator end, Axis inAxis = Axis::NONE)
        {
            switch (inAxis)
            {
            case Axis::NONE:
            {
                uint32 finalSize = 0;
                auto iter = begin;
                while (iter != end)
                {
                    const auto &ndarray = *iter++;
                    finalSize += ndarray.size();
                }

                Matrix<dtype> returnArray(1, finalSize);
                uint32 offset = 0;
                iter = begin;
                while (iter != end)
                {
                    const auto &ndarray = *iter++;
                    algo::copy(ndarray.cbegin(), ndarray.cend(), returnArray.begin() + offset);
                    offset += ndarray.size();
                }

                return returnArray;
            }
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
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }
    }

    /// Join a sequence of arrays along an existing axis.
    template <typename dtype>
    Matrix<dtype> concatenate(const std::initializer_list<Matrix<dtype>> &inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::concatenate<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }

    //============================================================================
    // Method Description:
    /// Join a sequence of arrays along an existing axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html
    ///
    /// @param inArrayList
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    ///
    template <typename dtype>
    Matrix<dtype> concatenate(const std::vector<Matrix<dtype>> &inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::concatenate<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }
}
