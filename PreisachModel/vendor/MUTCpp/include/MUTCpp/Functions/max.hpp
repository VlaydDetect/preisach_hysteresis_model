// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    template <typename T>
    T max(const T &left, const T &right)
    {
        return std::max(left, right);
    }

    template <typename Derived>
    auto max(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::reduce(input, axis, [](const auto &values)
        {
            if (values.size() == 0) throw std::invalid_argument("Empty reduction");
            return values.maxCoeff();
        });
    }
}
