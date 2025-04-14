// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Error.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Change elements of an array based on conditional and input values.
    /// @param arr: Array to put data into.
    /// @param mask: Boolean mask array. Must have the same size as arr
    /// @param vals: Values to put into arr. Only the first N elements are used, where N is the
    ///              number of True values in mask. If vals is smaller than N, it will be repeated.
    template <typename dtype>
    void place(Matrix<dtype> &arr, const Matrix<bool> &mask, const Matrix<dtype> &vals)
    {
        if (mask.size() != arr.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input arguments 'arr' and 'mask' must have the same size.");
        }

        if (vals.isempty())
        {
            return;
        }

        auto valIdx = 0;
        for (decltype(arr.size()) i = 0; i < arr.size(); ++i)
        {
            if (mask[i])
            {
                arr[i] = vals[valIdx++ % vals.size()];
            }
        }
    }
}
