// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"

namespace mc
{
    template <typename Left, typename Right>
    auto inner(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        if (left.size() != right.size()) throw std::invalid_argument("Inner product sizes differ");
        const auto a = detail::rowMajor(left);
        const auto b = detail::rowMajor(right);
        using Result = decltype(a(0) * b(0));
        return std::inner_product(a.begin(), a.end(), b.begin(), Result{});
    }
}
