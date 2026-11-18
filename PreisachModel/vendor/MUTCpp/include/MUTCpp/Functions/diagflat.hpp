// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cstdlib>

namespace mc
{
    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> diagflat(const Eigen::DenseBase<Derived> &input, Eigen::Index offset = 0)
    {
        using T = typename Derived::Scalar;
        const Eigen::Index size = input.size() + std::abs(offset);
        Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Zero(size, size);
        if (input.size() != 0) result.diagonal(offset) = detail::rowMajor(input);
        return result;
    }
}
