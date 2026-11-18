// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/concatenate.hpp"

namespace mc
{
    template <typename Derived>
    auto stack(const std::vector<Derived> &arrays, Axis axis = Axis::NONE)
    {
        if (axis != Axis::ROW && axis != Axis::COL)
        {
            throw std::invalid_argument("Stack axis must be ROW or COL");
        }
        return concatenate(arrays, axis);
    }

    template <typename Derived>
    auto stack(const std::initializer_list<Derived> &arrays, Axis axis = Axis::NONE)
    {
        if (axis != Axis::ROW && axis != Axis::COL)
        {
            throw std::invalid_argument("Stack axis must be ROW or COL");
        }
        return concatenate(arrays, axis);
    }
}
