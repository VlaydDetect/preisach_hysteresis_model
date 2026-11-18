// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/unique.hpp"

namespace mc
{
    template <typename Left, typename Right>
    Eigen::RowVectorX<typename Left::Scalar> union1d(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        const auto a = unique(left);
        const auto b = unique(right);
        Eigen::RowVectorX<typename Left::Scalar> values(a.size() + b.size());
        const auto end = std::set_union(a.begin(), a.end(), b.begin(), b.end(), values.begin());
        return values.head(std::distance(values.begin(), end));
    }
}
