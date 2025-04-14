// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <initializer_list>
#include <vector>

#include "Functions/row_stack.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the variance along the specified axis.
    ///
    template<typename dtype>
    Matrix<dtype> vstack(std::initializer_list<Matrix<dtype>> inArrayList)
    {
        return detail::row_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }

    /// Compute the variance along the specified axis.
    template<typename dtype>
    Matrix<dtype> vstack(std::vector<Matrix<dtype>> inArrayList)
    {
        return detail::row_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }
}
