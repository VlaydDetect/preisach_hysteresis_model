// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>

#include "Core/Error.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the elements of an array that satisfy some condition.
    /// @param condition: An array whose nonzero or True entries indicate the elements of arr to extract.
    /// @param arr: Input array of the same size as condition
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> extract(const Matrix<bool> &condition, const Matrix<dtype> &arr)
    {
        if (condition.size() != arr.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input arguments 'condition' and 'arr' must have the same size.");
        }

        std::vector<dtype> values;
        for (decltype(arr.size()) i = 0; i < arr.size(); ++i)
        {
            if (condition[i])
            {
                values.push_back(arr[i]);
            }
        }

        return Matrix<dtype>(values.begin(), values.end());
    }
}
