// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    template <typename T>
        requires (std::is_arithmetic_v<T> || mc::is_complex_v<T>)
    auto powerf(T value, double exponent)
    {
        return utils::powerf(value, exponent);
    }

    template <typename Derived>
    auto powerf(const Eigen::DenseBase<Derived> &input, double exponent)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::powerf(value, exponent);
        }).eval();
    }

    template <typename Left, typename Right>
    auto powerf(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::powerf(a, b);
        });
    }
}
