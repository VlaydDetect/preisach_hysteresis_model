// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace mc
{
    template <typename T, typename Function>
    Eigen::RowVectorX<T> fromfunction(Function function, Eigen::Index size)
    {
        if (size < 0) throw std::invalid_argument("Size cannot be negative");
        Eigen::RowVectorX<T> result(size);
        for (Eigen::Index i = 0; i < size; ++i) result(i) = function(i);
        return result;
    }

    template <typename T, typename Function>
    Eigen::MatrixX<T> fromfunction(Function function, Eigen::Index rows, Eigen::Index cols)
    {
        if (rows < 0 || cols < 0) throw std::invalid_argument("Dimensions cannot be negative");
        Eigen::MatrixX<T> result(rows, cols);
        for (Eigen::Index row = 0; row < rows; ++row)
        {
            for (Eigen::Index col = 0; col < cols; ++col) result(row, col) = function(row, col);
        }
        return result;
    }
}
