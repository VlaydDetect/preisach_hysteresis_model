// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <initializer_list>
#include <vector>

#include "Functions/column_stack.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Stack arrays in sequence horizontally (column wise).
    template<typename dtype>
    Matrix<dtype> hstack(std::initializer_list<Matrix<dtype>> inArrayList)
    {
        return detail::column_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }

    /// Stack arrays in sequence horizontally (column wise).
    template<typename dtype>
    Matrix<dtype> hstack(std::vector<Matrix<dtype>> inArrayList)
    {
        return detail::column_stack<dtype>(inArrayList.begin(), inArrayList.end());
    }
}
