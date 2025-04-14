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
#include "Matrix.hpp"

namespace mc
{
    namespace detail
    {
        /// Stack 1-D arrays as columns into a 2-D array.
        /// @param begin: iterator to the beginning of the span
        /// @param end: iterator to one past the end of the span
        /// @return Matrix
        template<typename dtype, typename Iterator>
        Matrix<dtype> column_stack(Iterator begin, Iterator end)
        {
            // first loop through to calculate the final size of the array
            Shape finalShape;
            auto  iter = begin;
            while (iter != end)
            {
                const auto& ndarray = *iter++;
                if (ndarray.shape().isnull())
                {
                    continue;
                }

                if (finalShape.isnull())
                {
                    finalShape = ndarray.shape();
                }
                else if (ndarray.shape().rows != finalShape.m_Rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input arrays must have the same number of rows.");
                }
                else
                {
                    finalShape.m_Cols += ndarray.shape().cols;
                }
            }

            // now that we know the final size, contruct the output array
            Matrix<dtype> returnArray(finalShape);
            uint32         colStart = 0;
            iter                    = begin;
            while (iter != end)
            {
                const auto& ndarray  = *iter++;
                const Shape theShape = ndarray.shape();
                for (uint32 row = 0; row < theShape.m_Rows; ++row)
                {
                    for (uint32 col = 0; col < theShape.m_Cols; ++col)
                    {
                        returnArray(row, colStart + col) = ndarray(row, col);
                    }
                }
                colStart += theShape.m_Cols;
            }

            return returnArray;
        }
    }

    /// Stack 1-D arrays as columns into a 2-D array.
    template<typename dtype>
    Matrix<dtype> column_stack(const std::initializer_list<Matrix<dtype>>& inArrayList)
    {
        return detail::column_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }

    /// Stack 1-D arrays as columns into a 2-D array.
    template<typename dtype>
    Matrix<dtype> column_stack(const std::vector<Matrix<dtype>>& inArrayList)
    {
        return detail::column_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }
}
