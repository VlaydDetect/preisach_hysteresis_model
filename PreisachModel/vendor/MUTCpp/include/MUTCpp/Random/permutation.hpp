// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/arange.hpp"
#include "Random/shuffle.hpp"

namespace mc::random
{
    template <typename T>
    requires std::is_integral_v<T>
    Eigen::RowVectorX<T> permutation(T count)
    {
        auto values = mc::arange(count);
        std::shuffle(values.begin(), values.end(), generator_);
        return values;
    }

    template <typename Derived>
    auto permutation(const Eigen::DenseBase<Derived> &input)
    {
        auto result = input.derived().eval();
        shuffle(result);
        return result;
    }
}
