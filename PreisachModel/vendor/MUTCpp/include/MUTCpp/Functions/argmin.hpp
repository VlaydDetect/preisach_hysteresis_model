// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    template <typename Derived>
    auto argmin(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::reduce(input, axis, [&](const auto &values)
        {
            if (values.size() == 0) throw std::invalid_argument("Empty index reduction");
            return std::distance(values.begin(), std::min_element(values.begin(), values.end()));
        });
    }
}
