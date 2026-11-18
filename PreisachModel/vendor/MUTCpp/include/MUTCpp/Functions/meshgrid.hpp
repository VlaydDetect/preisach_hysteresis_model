// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/arange.hpp"
#include "Utils/DenseOperations.hpp"
#include <utility>

namespace mc
{
    template <typename Left, typename Right>
    auto meshgrid(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        using T = std::common_type_t<typename Left::Scalar, typename Right::Scalar>;
        Eigen::MatrixX<T> x = detail::rowMajor(left).transpose().template cast<T>().replicate(right.size(), 1);
        Eigen::MatrixX<T> y = detail::rowMajor(right).template cast<T>().replicate(1, left.size());
        return std::pair(std::move(x), std::move(y));
    }
    template <typename T = Eigen::Index, typename XF, typename XS, typename XI, typename YF, typename YS, typename YI>
    auto meshgrid(const Eigen::ArithmeticSequence<XF, XS, XI> &x, const Eigen::ArithmeticSequence<YF, YS, YI> &y)
    {
        return meshgrid(arange<T>(x), arange<T>(y));
    }
}
