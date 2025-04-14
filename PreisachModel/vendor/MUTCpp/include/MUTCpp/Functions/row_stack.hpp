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
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace detail
    {
        /// Stack arrays in sequence vertically (row wise).
        /// @param begin: iterator to the beginning of the span
        /// @param end: iterator to one past the end of the span
        /// @return Matrix
        template<typename dtype, typename Iterator>
        Matrix<dtype> row_stack(Iterator begin, Iterator end)
        {
            // first loop through to calculate the final size of the array
            Shape finalShape;
            auto  iter = begin;
            while (iter != end)
            {
                const auto& matrix = *iter++;
                if (matrix.shape().isnull())
                {
                    continue;
                }

                if (finalShape.isnull())
                {
                    finalShape = matrix.shape();
                }
                else if (matrix.shape().m_Cols != finalShape.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input arrays must have the same number of columns.");
                }
                else
                {
                    finalShape.m_Rows += matrix.shape().m_Rows;
                }
            }

            // now that we know the final size, contruct the output array
            Matrix<dtype> returnArray(finalShape);
            uint32         rowStart = 0;
            iter                    = begin;
            while (iter != end)
            {
                const auto& ndarray  = *iter++;
                const Shape theShape = ndarray.shape();
                for (uint32 row = 0; row < theShape.m_Rows; ++row)
                {
                    for (uint32 col = 0; col < theShape.m_Cols; ++col)
                    {
                        returnArray(rowStart + row, col) = ndarray(row, col);
                    }
                }
                rowStart += theShape.m_Rows;
            }

            return returnArray;
        }
    }

    /// Stack arrays in sequence vertically (row wise).
    template<typename dtype>
    Matrix<dtype> row_stack(const std::initializer_list<Matrix<dtype>>& inArrayList)
    {
        return detail::row_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }

    /// Stack arrays in sequence vertically (row wise).
    template<typename dtype>
    Matrix<dtype> row_stack(const std::vector<Matrix<dtype>>& inArrayList)
    {
        return detail::row_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }
}
