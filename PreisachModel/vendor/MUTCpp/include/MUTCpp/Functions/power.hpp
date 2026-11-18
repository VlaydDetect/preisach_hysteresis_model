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
    auto power(T value, uint8 exponent)
    {
        return utils::power(value, exponent);
    }

    template <typename Derived>
    auto power(const Eigen::DenseBase<Derived> &input, uint8 exponent)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::power(value, exponent);
        }).eval();
    }

    template <typename Left, typename Right>
    auto power(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return mc::power(a, b);
        });
    }
}
