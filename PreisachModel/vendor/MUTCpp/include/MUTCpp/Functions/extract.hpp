// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/find.hpp"

namespace mc
{
    template <typename Mask, typename Derived>
    Eigen::RowVectorX<typename Derived::Scalar> extract(const Eigen::DenseBase<Mask> &mask, const Eigen::DenseBase<Derived> &input)
    {
        if (mask.size() != input.size()) throw std::invalid_argument("Input and mask sizes must match");
        return detail::rowMajor(input)(find(mask)).transpose().matrix();
    }
}
