// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cmath>

namespace mc
{
    template <typename T>
        requires std::is_arithmetic_v<T>
    T minimum(T left, T right)
    {
        return static_cast<T>(std::min(left, right));
    }

    template <typename Left, typename Right>
    auto minimum(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::minimum(a, static_cast<decltype(a)>(b));
        });
    }

    template <typename Derived>
    auto minimum(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar other)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::minimum(value, other);
        }).eval();
    }

    template <typename Derived>
    auto minimum(typename Derived::Scalar other, const Eigen::DenseBase<Derived> &input)
    {
        return minimum(input, other);
    }
}
