// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/argwhere.hpp"
#include <limits>

namespace mc
{
    template <typename Derived>
    requires std::is_same_v<typename Derived::Scalar, bool>
    Eigen::RowVectorX<Eigen::Index> find(const Eigen::DenseBase<Derived> &mask, Eigen::Index count = (std::numeric_limits<Eigen::Index>::max)())
    {
        if (count < 0) throw std::invalid_argument("Find count cannot be negative");
        const auto indices = argwhere(mask);
        return indices.head((std::min)(count, indices.size()));
    }
}
