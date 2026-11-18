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
    template <typename Left, typename Right>
    bool array_equiv(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        if (left.size() != right.size()) return false;
        const auto a = detail::rowMajor(left);
        const auto b = detail::rowMajor(right);
        return std::equal(a.begin(), a.end(), b.begin(), [](auto x, auto y)
        {
            if constexpr (std::is_integral_v<decltype(x)> && std::is_integral_v<decltype(y)>) return x == y;
            else return utils::essentiallyEqual(x, static_cast<decltype(x)>(y));
        });
    }
}
