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
    auto logical_xor(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return detail::broadcast(left, right, [](auto a, auto b)
        {
            return utils::essentiallyEqual(a, decltype(a){}) != utils::essentiallyEqual(b, decltype(b){});
        });
    }
}
