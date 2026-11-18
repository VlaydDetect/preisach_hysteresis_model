// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <Eigen/Geometry>

namespace mc
{
    template <typename Left, typename Right>
    Eigen::MatrixX<typename Left::Scalar> cross(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, Axis axis = Axis::NONE)
    {
        using T = typename Left::Scalar;
        const auto product = [](const auto &a, const auto &b) -> Eigen::RowVectorX<T>
        {
            if (a.size() != b.size() || (a.size() != 2 && a.size() != 3))
            {
                throw std::invalid_argument("Cross product needs two or three components");
            }
            if (a.size() == 2) return Eigen::RowVectorX<T>::Constant(1, a(0) * b(1) - a(1) * b(0));
            const Eigen::Vector3<T> first(a(0), a(1), a(2));
            const Eigen::Vector3<T> second(b(0), b(1), b(2));
            return first.cross(second).transpose();
        };
        if (axis == Axis::NONE) return product(detail::rowMajor(left), detail::rowMajor(right));
        if (axis != Axis::ROW && axis != Axis::COL) throw std::invalid_argument("Unsupported axis");
        if (left.rows() != right.rows() || left.cols() != right.cols())
        {
            throw std::invalid_argument("Cross product shape mismatch");
        }
        const Eigen::Index count = axis == Axis::COL ? left.rows() : left.cols();
        const Eigen::Index size = axis == Axis::COL ? left.cols() : left.rows();
        if (size != 2 && size != 3) throw std::invalid_argument("Cross product needs two or three components");
        const Eigen::Index components = size == 2 ? 1 : 3;
        Eigen::MatrixX<T> result(axis == Axis::COL ? count : components, axis == Axis::COL ? components : count);
        for (Eigen::Index i = 0; i < count; ++i)
        {
            if (axis == Axis::COL) result.row(i) = product(left.derived().row(i), right.derived().row(i));
            else result.col(i) = product(left.derived().col(i), right.derived().col(i)).transpose();
        }
        return result;
    }
}
