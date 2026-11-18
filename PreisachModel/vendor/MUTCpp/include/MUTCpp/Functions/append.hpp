// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/concatenate.hpp"

namespace mc
{
    template <typename Left, typename Right>
    Eigen::MatrixX<typename Left::Scalar> append(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, Axis axis = Axis::NONE)
    {
        using T = typename Left::Scalar;
        if (left.rows() == 0 && left.cols() == 0) return right.derived().matrix();
        if (right.rows() == 0 && right.cols() == 0) return left.derived().matrix();
        using Ref = Eigen::Ref<const Eigen::MatrixX<T>>;
        return concatenate({Ref(left.derived().matrix()), Ref(right.derived().matrix())}, axis);
    }
}
