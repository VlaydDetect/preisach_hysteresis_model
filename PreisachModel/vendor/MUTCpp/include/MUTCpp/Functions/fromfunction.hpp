// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Construct an array by executing a function over each coordinate.
    /// The resulting array therefore has a value fn(x) at coordinate(x).
    ///  @param func: callable that accepts an integer coordinate and returns type T
    ///  @param size: the size of the 1d array to create
    ///  @return Matrix
    template <typename dtype>
    Matrix<dtype> fromfunction(const std::function<dtype(typename Matrix<dtype>::size_type)> func,
                               typename Matrix<dtype>::size_type size)
    {
        Matrix<dtype> result(1, size);
        const auto indices = [size]
        {
            std::vector<typename Matrix<dtype>::size_type> temp(size);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();

        algo::transform(indices.begin(),
                        indices.end(),
                        result.begin(),
                        [&func](const auto idx) { return func(idx); });

        return result;
    }

    /// Construct an array by executing a function over each coordinate.
    /// The resulting array therefore has a value fn(x, y) at coordinate(x, y).
    ///  @param func: callable that accepts an integer coordinate and returns type T
    ///  @param shape: the shape of the array to create
    ///  @return Matrix
    template <typename dtype>
    Matrix<dtype> fromfunction(
        const std::function<dtype(typename Matrix<dtype>::size_type, typename Matrix<dtype>::size_type)> func,
        Shape shape)
    {
        Matrix<dtype> result(shape);
        const auto rows = [&shape]
        {
            std::vector<typename Matrix<dtype>::size_type> temp(shape.m_Rows);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();
        const auto cols = [&shape]
        {
            std::vector<typename Matrix<dtype>::size_type> temp(shape.m_Cols);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();

        algo::for_each(rows.begin(),
                       rows.end(),
                       [&cols, &result, &func](const auto row)
                       {
                           algo::transform(cols.begin(),
                                           cols.end(),
                                           result.begin(row),
                                           [&func, row](const auto col) { return func(row, col); });
                       });

        return result;
    }
}
