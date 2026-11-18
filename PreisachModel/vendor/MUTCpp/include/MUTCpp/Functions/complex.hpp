// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <complex>

namespace mc
{
    template <typename T>
    std::complex<T> complex(T real, T imaginary = T{})
    {
        return {real, imaginary};
    }

    template <typename Derived>
    auto complex(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return std::complex<typename Derived::Scalar>(value);
        }).eval();
    }

    template <typename Left, typename Right>
    auto complex(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return std::complex<decltype(a)>(a, b);
        });
    }
}
