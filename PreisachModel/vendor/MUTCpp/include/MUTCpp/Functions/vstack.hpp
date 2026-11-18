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
    auto vstack(const std::vector<Derived> &arrays)
    {
        return concatenate(arrays, Axis::ROW);
    }

    template <typename Derived>
    auto vstack(const std::initializer_list<Derived> &arrays)
    {
        return concatenate(arrays, Axis::ROW);
    }
}
