// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <iterator>

#include "Core/Enums.hpp"
#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Find indices where elements should be inserted to maintain order.
    /// Find the indices into a sorted array a such that, if the corresponding elements in v were
    /// inserted before the indices, the order of a would be preserved.
    /// Assuming that an is sorted:
    ///    side     returned index i satisfies
    ///    left     a[i - 1] < v ⇐ a[i]
    ///    right    a[i - 1] ⇐ v < a[i]
    ///
    /// @param inArray: the input array
    /// @param inValue: values to insert into a
    /// @param side: If ‘left’, the index of the first suitable location found is given. If ‘right’, return the
    ///              last such index. If there is no suitable index, return either 0 or N (where N is the length of a).
    /// @return Array of insertion points with the same shape as v, or an integer if v is a scalar.
    template <typename dtype>
    typename Matrix<dtype>::index_type searchsorted(const Matrix<dtype> &inArray, dtype inValue, Side side = Side::LEFT)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        switch (side)
        {
        case Side::LEFT:
        {
            return std::distance(inArray.begin(), std::lower_bound(inArray.begin(), inArray.end(), inValue));
        }
        case Side::RIGHT:
        {
            return std::distance(inArray.begin(), std::upper_bound(inArray.begin(), inArray.end(), inValue));
        }
        }

        // get rid of compiler warning
        return {};
    }

    /// Find indices where elements should be inserted to maintain order.
    /// Find the indices into a sorted array a such that, if the corresponding elements in v were
    /// inserted before the indices, the order of a would be preserved.
    /// Assuming that an is sorted:
    ///    side     returned index i satisfies
    ///    left     a[i - 1] < v ⇐ a[i]
    ///    right    a[i - 1] ⇐ v < a[i]
    ///    
    /// @param inArray: the input array
    /// @param inValues: values to insert into a
    /// @param side: If ‘left’, the index of the first suitable location found is given. If ‘right’, return the
    ///              last such index. If there is no suitable index, return either 0 or N (where N is the length of a).
    /// @return Array of insertion points with the same shape as v, or an integer if v is a scalar.
    template <typename dtype>
    Matrix<typename Matrix<dtype>::index_type> searchsorted(const Matrix<dtype> &inArray, const Matrix<dtype> &inValues,
                                                            Side side = Side::LEFT)
    {
        Matrix<typename Matrix<dtype>::index_type> indices(1, inValues.size());
        std::transform(inValues.begin(),
                       inValues.end(),
                       indices.begin(),
                       [&inArray, side](const auto &value) { return searchsorted(inArray, value, side); });
        return indices;
    }
}
