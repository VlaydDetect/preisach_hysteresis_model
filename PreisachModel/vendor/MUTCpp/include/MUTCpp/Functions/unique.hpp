// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::RowVectorX<typename Derived::Scalar> unique(const Eigen::DenseBase<Derived> &input)
    {
        using T = typename Derived::Scalar;
        Eigen::RowVectorX<T> values = detail::rowMajor(input).transpose();
        std::sort(values.begin(), values.end());
        const auto end = std::unique(values.begin(), values.end(), [](const T &a, const T &b)
        {
            return utils::essentiallyEqual(a, b);
        });
        return values.head(std::distance(values.begin(), end));
    }
}
