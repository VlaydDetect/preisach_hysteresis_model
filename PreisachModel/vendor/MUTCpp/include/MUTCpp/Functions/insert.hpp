// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Slice.hpp"
#include "Core/Types.hpp"
#include "Functions/ones_like.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Insert values before the given indices.
    /// @param arr: input array.
    /// @param index: index to insert the value before in the flattened
    /// @param value: value to insert
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, int32 index, const dtype &value)
    {
        const Matrix<dtype> values = {value};
        return insert(arr, index, values);
    }

    /// Insert values before the given indices.
    /// @param arr: input array.
    /// @param index: index to insert the values before in the flattened
    /// @param values: value to insert
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, int32 index, const Matrix<dtype> &values)
    {
        if (index < 0)
        {
            index += arr.size();
            // still
            if (index < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("index out of range");
            }
        }
        else if (index > static_cast<int32>(arr.size()))
        {
            THROW_INVALID_ARGUMENT_ERROR("index out of range");
        }

        const auto valuesSlice = Slice(index, index + values.size());
        auto result = Matrix<dtype>(1, arr.size() + values.size());
        result.put(valuesSlice, values);

        Matrix<bool> mask(result.shape());
        mask.fill(true);
        mask.put(valuesSlice, false);
        result.putMask(mask, arr.flatten());

        return result;
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param index: index to insert the values before
    /// @param value: value to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, int32 index, const dtype &value, Axis axis)
    {
        const Matrix<dtype> values = {value};
        return insert(arr, index, values, axis);
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param index: index to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, int32 index, const Matrix<dtype> &values, Axis axis)
    {
        switch (axis)
        {
        case Axis::NONE:
        {
            return insert(arr, index, values);
        }
        case Axis::ROW:
        {
            if (!(values.size() == arr.numCols() || values.isscalar() || values.numCols() == arr.numCols()))
            {
                THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
            }

            if (index < 0)
            {
                index += arr.numRows();
                // still
                if (index < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("index out of range");
                }
            }
            else if (index > static_cast<int32>(arr.numRows()))
            {
                THROW_INVALID_ARGUMENT_ERROR("index out of range");
            }

            auto result = Matrix<dtype>();
            int32 valuesSize{};
            if (values.size() == arr.numCols() || values.isscalar())
            {
                result.resizeFast(arr.numRows() + 1, arr.numCols());
                valuesSize = 1;
            }
            else if (values.numCols() == arr.numCols())
            {
                result.resizeFast(arr.numRows() + values.numRows(), arr.numCols());
                valuesSize = values.numRows();
            }

            auto mask = ones_like<bool>(result);
            mask.put(Slice(index, index + valuesSize), mask.cSlice(), false);

            result.putMask(mask, arr);
            result.putMask(!mask, values);

            return result;
        }
        case Axis::COL:
        {
            if (!(values.size() == arr.numRows() || values.isscalar() || values.numRows() == arr.numRows()))
            {
                THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
            }

            if (index < 0)
            {
                index += arr.numCols();
                // still
                if (index < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("index out of range");
                }
            }
            else if (index > static_cast<int32>(arr.numCols()))
            {
                THROW_INVALID_ARGUMENT_ERROR("index out of range");
            }

            auto result = Matrix<dtype>();
            int32 valuesSize{};
            if (values.size() == arr.numRows() || values.isscalar())
            {
                result.resizeFast(arr.numRows(), arr.numCols() + 1);
                valuesSize = 1;
            }
            else if (values.numRows() == arr.numRows())
            {
                result.resizeFast(arr.numRows(), arr.numCols() + values.numCols());
                valuesSize = values.numCols();
            }

            auto mask = ones_like<bool>(result);
            mask.put(mask.rSlice(), Slice(index, index + valuesSize), false);

            result.putMask(mask, arr);
            result.putMask(!mask, values);

            return result;
        }
        default:
        {
            // get rid of compiler warning
            return {};
        }
        }
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param indices: indices to insert the values before
    /// @param value: value to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    Matrix<dtype> insert(const Matrix<dtype> &arr, const Indices &indices, const dtype &value, Axis axis = Axis::NONE)
    {
        const Matrix<dtype> values = {value};
        return insert(arr, indices, values, axis);
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param slice: slice to insert the values before
    /// @param value: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, Slice slice, const dtype &value, Axis axis = Axis::NONE)
    {
        auto sliceIndices = slice.toIndices(arr.dimSize(axis));
        return insert(arr,
                      Matrix<uint32>(sliceIndices.data(), sliceIndices.size(), PointerPolicy::SHELL),
                      value,
                      axis);
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param indices: indices to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    Matrix<dtype>
    insert(const Matrix<dtype> &arr, const Indices &indices, const Matrix<dtype> &values, Axis axis = Axis::NONE)
    {
        const auto isScalarValue = values.isscalar();

        switch (axis)
        {
        case Axis::NONE:
        {
            if (!isScalarValue && indices.size() != values.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("could not broadcast values into indices");
            }

            const auto arrSize = static_cast<int32>(arr.size());

            std::vector<std::pair<int32, dtype>> indexValues(indices.size());
            if (isScalarValue)
            {
                const auto value = values.front();
                algo::transform(indices.begin(),
                                indices.end(),
                                indexValues.begin(),
                                [arrSize, value](auto index) -> std::pair<int32, dtype>
                                {
                                    if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                                    {
                                        if (index < 0)
                                        {
                                            index += arrSize;
                                        }
                                        // still
                                        if (index < 0)
                                        {
                                            THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                        }
                                    }
                                    if (static_cast<int32>(index) > arrSize)
                                    {
                                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                    }

                                    return std::make_pair(static_cast<int32>(index), value);
                                });
            }
            else
            {
                algo::transform(indices.begin(),
                                indices.end(),
                                values.begin(),
                                indexValues.begin(),
                                [arrSize](auto index, const auto &value) -> std::pair<int32, dtype>
                                {
                                    if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                                    {
                                        if (index < 0)
                                        {
                                            index += arrSize;
                                        }
                                        // still
                                        if (index < 0)
                                        {
                                            THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                        }
                                    }
                                    if (static_cast<int32>(index) > arrSize)
                                    {
                                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                    }

                                    return std::make_pair(static_cast<int32>(index), value);
                                });
            }

            algo::sort(indexValues.begin(),
                       indexValues.end(),
                       [](const auto &indexValue1, const auto &indexValue2) noexcept -> bool
                       {
                           return indexValue1.first < indexValue2.first;
                       });
            auto indexValuesUnique = std::vector<typename decltype(indexValues)::value_type>{};
            std::unique_copy(indexValues.begin(),
                             indexValues.end(),
                             std::back_inserter(indexValuesUnique),
                             [](const auto &indexValue1, const auto &indexValue2) noexcept -> bool
                             {
                                 return indexValue1.first == indexValue2.first;
                             });

            auto result = Matrix<dtype>(1, arr.size() + indexValuesUnique.size());

            auto mask = ones_like<bool>(result);
            int32 counter = 0;
            std::for_each(indexValuesUnique.begin(),
                          indexValuesUnique.end(),
                          [&counter, &mask](auto &indexValue) noexcept -> void
                          {
                              mask[indexValue.first + counter++] = false;
                          });

            result.putMask(mask, arr);

            auto valuesSorted = [&indexValuesUnique]
            {
                std::vector<dtype> values_;
                values_.reserve(indexValuesUnique.size());
                std::transform(indexValuesUnique.begin(),
                               indexValuesUnique.end(),
                               std::back_inserter(values_),
                               [](const auto &indexValue) { return indexValue.second; });
                return values_;
            }();

            result.putMask(!mask, Matrix<dtype>(valuesSorted.data(), valuesSorted.size(), PointerPolicy::SHELL));

            return result;
        }
        case Axis::ROW:
        {
            const auto arrNumRows = static_cast<int32>(arr.numRows());

            std::vector<std::pair<int32, Matrix<dtype>>> indexValues(indices.size());
            if (values.isscalar())
            {
                const auto value = values.front();
                auto valueRow = Matrix<dtype>(1, arr.numCols());
                valueRow.fill(value);
                algo::transform(indices.begin(),
                                indices.end(),
                                indexValues.begin(),
                                [arrNumRows, &valueRow](auto index) -> std::pair<int32, Matrix<dtype>>
                                {
                                    if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                                    {
                                        if (index < 0)
                                        {
                                            index += arrNumRows;
                                        }
                                        // still
                                        if (index < 0)
                                        {
                                            THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                        }
                                    }
                                    if (static_cast<int32>(index) > arrNumRows)
                                    {
                                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                    }

                                    return std::make_pair(static_cast<int32>(index), valueRow);
                                });
            }
            else if (values.size() == arr.numCols())
            {
                algo::transform(indices.begin(),
                                indices.end(),
                                indexValues.begin(),
                                [arrNumRows, &values](auto index) -> std::pair<int32, Matrix<dtype>>
                                {
                                    if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                                    {
                                        if (index < 0)
                                        {
                                            index += arrNumRows;
                                        }
                                        // still
                                        if (index < 0)
                                        {
                                            THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                        }
                                    }
                                    if (static_cast<int32>(index) > arrNumRows)
                                    {
                                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                    }

                                    return std::make_pair(static_cast<int32>(index), values);
                                });
            }
            else if (values.numCols() == arr.numCols() && values.numRows() == indices.size())
            {
                int32 counter = 0;
                std::transform(indices.begin(),
                               indices.end(),
                               indexValues.begin(),
                               [arrNumRows, &values, &counter](auto index) -> std::pair<int32, Matrix<dtype>>
                               {
                                   if constexpr (type_traits::is_matrix_signed_int_v<Indices>)
                                   {
                                       if (index < 0)
                                       {
                                           index += arrNumRows;
                                       }
                                       // still
                                       if (index < 0)
                                       {
                                           THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                       }
                                   }
                                   if (static_cast<int32>(index) > arrNumRows)
                                   {
                                       THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                   }

                                   return std::make_pair(static_cast<int32>(index),
                                                         values(counter++, values.cSlice()));
                               });
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
            }

            algo::sort(indexValues.begin(),
                       indexValues.end(),
                       [](const auto &indexValue1, const auto &indexValue2) noexcept -> bool
                       {
                           return indexValue1.first < indexValue2.first;
                       });
            auto indexValuesUnique = std::vector<typename decltype(indexValues)::value_type>{};
            std::unique_copy(indexValues.begin(),
                             indexValues.end(),
                             std::back_inserter(indexValuesUnique),
                             [](const auto &indexValue1, const auto &indexValue2) noexcept -> bool
                             {
                                 return indexValue1.first == indexValue2.first;
                             });

            auto result = Matrix<dtype>(arrNumRows + indexValuesUnique.size(), arr.numCols());

            auto mask = ones_like<bool>(result);
            int32 counter = 0;
            std::for_each(indexValuesUnique.begin(),
                          indexValuesUnique.end(),
                          [&counter, &mask](auto &indexValue) noexcept -> void
                          {
                              mask.put(indexValue.first + counter++, mask.cSlice(), false);
                          });

            result.putMask(mask, arr);

            counter = 0;
            for (const auto &[index, values_] : indexValuesUnique)
            {
                result.put(index + counter++, result.cSlice(), values_);
            }

            return result;
        }
        case Axis::COL:
        {
            return insert(arr.transpose(), indices, values.transpose(), Axis::ROW).transpose();
        }
        default:
        {
            // get rid of compiler warning
            return {};
        }
        }
    }

    /// Insert values along the given axis before the given indices.
    /// @param arr: input array.
    /// @param slice: slice to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    template <typename dtype>
    Matrix<dtype> insert(const Matrix<dtype> &arr, Slice slice, const Matrix<dtype> &values, Axis axis = Axis::NONE)
    {
        auto sliceIndices = slice.toIndices(arr.dimSize(axis));
        return insert(arr,
                      Matrix<uint32>(sliceIndices.data(), sliceIndices.size(), PointerPolicy::SHELL),
                      values,
                      axis);
    }
}
