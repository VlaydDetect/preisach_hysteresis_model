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
    T gcd(T left, T right)
    {
        return static_cast<T>(std::gcd(left, right));
    }

    template <typename Left, typename Right>
    auto gcd(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::gcd(a, static_cast<decltype(a)>(b));
        });
    }

    template <typename Derived>
    typename Derived::Scalar gcd(const Eigen::DenseBase<Derived> &input)
    {
        using T = typename Derived::Scalar;
        const auto values = detail::rowMajor(input);
        return std::accumulate(values.begin(), values.end(), T{0},
            [](T left, T right) { return std::gcd(left, right); });
    }
}
