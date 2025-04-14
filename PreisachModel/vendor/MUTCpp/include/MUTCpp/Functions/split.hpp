// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>

#include "unique.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Split an array into multiple subarrays as views into an array.
    /// @param inArray
    /// @param indices: the indices to split
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    std::vector<Matrix<dtype>> split(const Matrix<dtype> &inArray, const Indices &indices, Axis inAxis = Axis::ROW)
    {
        switch (inAxis)
        {
        case Axis::ROW:
        {
            return vsplit(inArray, indices);
        }
        case Axis::COL:
        {
            return hsplit(inArray, indices);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("input inAxis must be either Axis::ROW or Axis::COL");
        }
        }

        return {}; // get rid of compiler warning
    }

    /// Split an array into multiple subarrays vertically (row-wise).
    /// @param inArray
    /// @param indices: the indices to split
    /// @return Matrix
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    std::vector<Matrix<dtype>> vsplit(const Matrix<dtype> &inArray, const Indices &indices)
    {
        const auto numRows = static_cast<int32>(inArray.numRows());
        Matrix<int32> uniqueIndices(1, indices.size());
        algo::transform(indices.begin(),
                        indices.end(),
                        uniqueIndices.begin(),
                        [numRows](auto index) noexcept -> int32
                        {
                            if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                            {
                                if (index < 0)
                                {
                                    index = std::max(index + numRows, int32{0});
                                }
                            }
                            if (index > numRows - 1)
                            {
                                index = numRows - 1;
                            }

                            return static_cast<int32>(index);
                        });
        uniqueIndices = unique(uniqueIndices);

        std::vector<Matrix<dtype>> splits{};
        splits.reserve(uniqueIndices.size() + 1);

        const auto cSlice = inArray.cSlice();
        int32 lowerIdx = 0;
        for (const auto index : uniqueIndices)
        {
            if (index == 0)
            {
                splits.push_back(Matrix<dtype>(Shape(0, inArray.numCols())));
                continue;
            }
            splits.push_back(inArray(Slice(lowerIdx, index), cSlice));

            lowerIdx = index;
        }

        if (lowerIdx < numRows - 1)
        {
            splits.push_back(inArray(Slice(lowerIdx, numRows), cSlice));
        }
        else
        {
            splits.push_back(inArray(-1, cSlice));
        }

        return splits;
    }

    /// Split an array into multiple subarrays horizontal (column-wise).
    /// @param inArray
    /// @param indices: the indices to split
    /// @return Matrix
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    std::vector<Matrix<dtype>> hsplit(const Matrix<dtype> &inArray, const Indices &indices)
    {
        const auto numCols = static_cast<int32>(inArray.numCols());
        Matrix<int32> uniqueIndices(1, indices.size());
        algo::transform(indices.begin(),
                        indices.end(),
                        uniqueIndices.begin(),
                        [numCols](auto index) noexcept -> int32
                        {
                            if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                            {
                                if (index < 0)
                                {
                                    index = std::max(index + numCols, int32{0});
                                }
                            }
                            if (static_cast<int32>(index) > numCols - 1)
                            {
                                index = numCols - 1;
                            }

                            return static_cast<int32>(index);
                        });
        uniqueIndices = unique(uniqueIndices);

        std::vector<Matrix<dtype>> splits{};
        splits.reserve(uniqueIndices.size() + 1);

        const auto rSlice = inArray.rSlice();
        int32 lowerIdx = 0;
        for (const auto index : uniqueIndices)
        {
            if (index == 0)

            {
                splits.push_back(Matrix<dtype>(Shape(inArray.numRows(), 0)));
                continue;
            }
            splits.push_back(inArray(rSlice, Slice(lowerIdx, index)));

            lowerIdx = index;
        }

        if (lowerIdx < numCols - 1)
        {
            splits.push_back(inArray(rSlice, Slice(lowerIdx, numCols)));
        }
        else
        {
            splits.push_back(inArray(rSlice, -1));
        }

        return splits;
    }
}
