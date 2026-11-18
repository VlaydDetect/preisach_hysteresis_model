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
    auto outer(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return (detail::rowMajor(left).matrix() * detail::rowMajor(right).matrix().transpose()).eval();
    }
}
