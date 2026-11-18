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
    T fmax(T left, T right)
    {
        return static_cast<T>(std::fmax(left, right));
    }

    template <typename Left, typename Right>
    auto fmax(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::fmax(a, static_cast<decltype(a)>(b));
        });
    }

    template <typename Derived>
    auto fmax(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar other)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::fmax(value, other);
        }).eval();
    }

    template <typename Derived>
    auto fmax(typename Derived::Scalar other, const Eigen::DenseBase<Derived> &input)
    {
        return fmax(input, other);
    }
}
