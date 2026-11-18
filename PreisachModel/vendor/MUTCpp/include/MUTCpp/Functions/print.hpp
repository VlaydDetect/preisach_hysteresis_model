// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <iostream>
#include "Utils/value2str.hpp"

namespace mc
{
    template <typename Derived>
    void print(const Eigen::DenseBase<Derived> &input)
    {
        std::cout << '[';
        for (Eigen::Index row = 0; row < input.rows(); ++row)
        {
            if (row != 0) std::cout << '\n';
            std::cout << '[';
            for (Eigen::Index col = 0; col < input.cols(); ++col)
            {
                std::cout << utils::value2str(input(row, col)) << ", ";
            }
            std::cout << ']';
        }
        std::cout << "]\n";
    }
}
