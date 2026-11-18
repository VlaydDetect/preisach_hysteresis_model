// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Core/Enums.hpp"
#include "Core/Types.hpp"
#include <cmath>
#include <limits>

namespace mc
{
    template <typename Derived>
    auto vander(const Eigen::DenseBase<Derived> &input, Eigen::Index count, Increasing increasing = Increasing::YES)
    {
        using T = decltype(std::pow(std::declval<typename Derived::Scalar>(), uint32{}));
        if (count < 0 || count > (std::numeric_limits<uint32>::max)())
        {
            throw std::invalid_argument("Invalid number of Vandermonde columns");
        }
        const auto values = detail::rowMajor(input);
        Eigen::MatrixX<T> result(values.size(), count);
        for (Eigen::Index col = 0; col < count; ++col)
        {
            const auto power = static_cast<uint32>(increasing == Increasing::YES ? col : count - col - 1);
            result.col(col) = values.unaryExpr([power](const auto &value) { return std::pow(value, power); });
        }
        return result;
    }

    template <typename Derived>
    auto vander(const Eigen::DenseBase<Derived> &input, Increasing increasing = Increasing::YES)
    {
        return vander(input, input.size(), increasing);
    }
}
