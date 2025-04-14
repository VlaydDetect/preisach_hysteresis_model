// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>
#include <vector>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Slice.hpp"
#include "Functions/unique.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace detail
    {
        /// Return a new array with subarrays deleted.
        template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        Matrix<dtype> deleteFlatIndices(const Matrix<dtype> &inArray, Indices inIndices)
        {
            if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
            {
                const auto arraySize = inArray.size();
                algo::for_each(inIndices.begin(),
                               inIndices.end(),
                               [arraySize](auto &value)
                               {
                                   if (value < 0)
                                   {
                                       value += arraySize;
                                   }
                               });
            }

            auto indices = unique(inIndices);

            std::vector<dtype> values;
            values.reserve(indices.size());
            for (int32 i = 0; i < static_cast<int32>(inArray.size()); ++i)
            {
                if (std::binary_search(indices.begin(), indices.end(), i))
                {
                    continue;
                }

                values.push_back(inArray[i]);
            }

            return Matrix<dtype>(values);
        }
        
        /// Return a new array with subarrays along the row axis deleted.
        template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        Matrix<dtype> deleteRowIndices(const Matrix<dtype> &inArray, Indices inIndices)
        {
            const auto arrayRows = static_cast<int32>(inArray.numRows());
            if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
            {
                algo::for_each(inIndices.begin(),
                               inIndices.end(),
                               [arrayRows](auto &value)
                               {
                                   if (value < 0)
                                   {
                                       value += arrayRows;
                                   }
                               });
            }

            auto indices = unique(inIndices);

            uint32 indicesSize = 0;
            std::for_each(indices.begin(),
                          indices.end(),
                          [arrayRows, &indicesSize](const auto &value)
                          {
                              if constexpr (std::is_signed_v<decltype(value)>)
                              {
                                  if (value >= 0 && value < arrayRows)
                                  {
                                      ++indicesSize;
                                  }
                              }
                              else
                              {
                                  if (value < arrayRows)
                                  {
                                      ++indicesSize;
                                  }
                              }
                          });

            const auto arrayCols = static_cast<int32>(inArray.numCols());
            Matrix<dtype> returnArray(arrayRows - indicesSize, arrayCols);

            uint32 rowCounter = 0;
            for (int32 row = 0; row < arrayRows; ++row)
            {
                if (std::binary_search(indices.begin(), indices.end(), row))
                {
                    continue;
                }

                for (int32 col = 0; col < arrayCols; ++col)
                {
                    returnArray(rowCounter, col) = inArray(row, col);
                }

                ++rowCounter;
            }

            return returnArray;
        }
        
        /// Return a new array with subarrays along the col axis deleted.
        template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        Matrix<dtype> deleteColumnIndices(const Matrix<dtype> &inArray, Indices inIndices)
        {
            const auto arrayCols = static_cast<int32>(inArray.numCols());
            if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
            {
                algo::for_each(inIndices.begin(),
                               inIndices.end(),
                               [arrayCols](auto &value)
                               {
                                   if (value < 0)
                                   {
                                       value += arrayCols;
                                   }
                               });
            }

            auto indices = unique(inIndices);

            uint32 indicesSize = 0;
            std::for_each(indices.begin(),
                          indices.end(),
                          [arrayCols, &indicesSize](const auto &value)
                          {
                              if constexpr (std::is_signed_v<decltype(value)>)
                              {
                                  if (value >= 0 && value < arrayCols)
                                  {
                                      ++indicesSize;
                                  }
                              }
                              else
                              {
                                  if (value < arrayCols)
                                  {
                                      ++indicesSize;
                                  }
                              }
                          });

            const auto arrayRows = static_cast<int32>(inArray.numRows());
            Matrix<dtype> returnArray(arrayRows, arrayCols - indicesSize);

            uint32 colCounter = 0;
            for (int32 col = 0; col < arrayCols; ++col)
            {
                if (std::binary_search(indices.begin(), indices.end(), col))
                {
                    continue;
                }

                for (int32 row = 0; row < arrayRows; ++row)
                {
                    returnArray(row, colCounter) = inArray(row, col);
                }

                ++colCounter;
            }

            return returnArray;
        }
    }

    /// Return a new array with subarrays along an axis deleted.
    /// @param inArray
    /// @param inIndices
    /// @param inAxis (Optional, default NONE) if NONE the indices will be applied to the flattened array
    /// @return Matrix
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    Matrix<dtype> deleteIndices(const Matrix<dtype> &inArray, const Indices &inIndices, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
        case Axis::NONE:
        {
            return detail::deleteFlatIndices(inArray, inIndices);
        }
        case Axis::ROW:
        {
            return detail::deleteRowIndices(inArray, inIndices);
        }
        case Axis::COL:
        {
            return detail::deleteColumnIndices(inArray, inIndices);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
    
    /// Return a new array with subarrays along an axis deleted.
    /// @param inArray
    /// @param inIndicesSlice
    /// @param inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> deleteIndices(const Matrix<dtype> &inArray, Slice inIndicesSlice, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
        case Axis::NONE:
        {
            inIndicesSlice.makePositiveAndValidate(inArray.size());
            break;
        }
        case Axis::ROW:
        {
            inIndicesSlice.makePositiveAndValidate(inArray.numRows());
            break;
        }
        case Axis::COL:
        {
            inIndicesSlice.makePositiveAndValidate(inArray.numCols());
            break;
        }
        }

        std::vector<int32> indices;
        for (auto i = inIndicesSlice.m_Start; i < inIndicesSlice.m_Stop; i += inIndicesSlice.m_Step)
        {
            indices.push_back(i);
        }

        return deleteIndices(inArray, Matrix<int32>(indices.data(), indices.size(), PointerPolicy::SHELL), inAxis);
    }
    
    /// Return a new array with subarrays along an axis deleted.
    /// @param inArray
    /// @param inIndex
    /// @param inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> deleteIndices(const Matrix<dtype> &inArray, int32 inIndex, Axis inAxis = Axis::NONE)
    {
        Matrix<int32> inIndices = {inIndex};
        return deleteIndices(inArray, inIndices, inAxis);
    }
}
