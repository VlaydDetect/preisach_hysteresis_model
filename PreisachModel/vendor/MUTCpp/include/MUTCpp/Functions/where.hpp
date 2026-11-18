// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <type_traits>

namespace mc
{
    template <typename Mask, typename Left, typename Right>
    Eigen::MatrixX<typename Left::Scalar> where(const Eigen::DenseBase<Mask> &mask, const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        if (mask.rows() != left.rows() || mask.cols() != left.cols() ||
            mask.rows() != right.rows() || mask.cols() != right.cols())
        {
            throw std::invalid_argument("Mask and input shapes must match");
        }
        return mask.select(left.derived(), right.derived()).matrix();
    }

    template <typename Mask, typename Left>
    auto where(const Eigen::DenseBase<Mask> &mask, const Eigen::DenseBase<Left> &left, typename Left::Scalar right)
    {
        return where(mask, left, Left::PlainObject::Constant(left.rows(), left.cols(), right));
    }

    template <typename Mask, typename Right>
    auto where(const Eigen::DenseBase<Mask> &mask, typename Right::Scalar left, const Eigen::DenseBase<Right> &right)
    {
        return where(mask, Right::PlainObject::Constant(right.rows(), right.cols(), left), right);
    }

    template <typename Mask, typename T>
    requires (!std::is_base_of_v<Eigen::EigenBase<T>, T>)
    Eigen::MatrixX<T> where(const Eigen::DenseBase<Mask> &mask, T left, T right)
    {
        return where(mask, Eigen::MatrixX<T>::Constant(mask.rows(), mask.cols(), left),
                           Eigen::MatrixX<T>::Constant(mask.rows(), mask.cols(), right));
    }
}
