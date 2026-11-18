// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Core/Enums.hpp"

namespace mc
{
    template <typename Derived, typename T>
    requires (!std::is_base_of_v<Eigen::EigenBase<T>, T>)
    Eigen::Index searchsorted(const Eigen::DenseBase<Derived> &input, const T &value, Side side = Side::LEFT)
    {
        const auto values = detail::rowMajor(input);
        const auto position = side == Side::LEFT
            ? std::lower_bound(values.begin(), values.end(), value)
            : std::upper_bound(values.begin(), values.end(), value);
        return std::distance(values.begin(), position);
    }

    template <typename Derived, typename Values>
    Eigen::RowVectorX<Eigen::Index> searchsorted(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Values> &values, Side side = Side::LEFT)
    {
        return detail::rowMajor(values).transpose().unaryExpr([&](const auto &value)
        {
            return searchsorted(input, value, side);
        });
    }
}
