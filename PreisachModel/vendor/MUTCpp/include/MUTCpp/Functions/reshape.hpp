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
    template <typename T>
    Eigen::MatrixX<T> &reshape(Eigen::MatrixX<T> &input, Eigen::Index rows, Eigen::Index columns)
    {
        if (rows < 0 && columns > 0) rows = input.size() / columns;
        if (columns < 0 && rows > 0) columns = input.size() / rows;
        if (rows < 0 || columns < 0 || (rows == 0 ? input.size() != 0 :
            input.size() / rows != columns || input.size() % rows != 0))
        {
            throw std::invalid_argument("Requested reshape changes element count");
        }
        input = input.template reshaped<Eigen::RowMajor>(rows, columns).eval();
        return input;
    }

    template <typename T>
    Eigen::MatrixX<T> &reshape(Eigen::MatrixX<T> &input, Eigen::Index count)
    {
        return reshape(input, 1, count);
    }
}
