// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Core/Types.hpp"

namespace mc
{
    template <typename T>
    Eigen::MatrixX<T> tril(uint32 rows, uint32 cols, int32 offset = 0)
    {
        return Eigen::MatrixX<T>::NullaryExpr(rows, cols, [offset](Eigen::Index row, Eigen::Index col)
        {
            return static_cast<T>(col - row <= offset);
        });
    }

    template <typename T>
    Eigen::MatrixX<T> tril(uint32 size, int32 offset = 0)
    {
        return tril<T>(size, size, offset);
    }

    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> tril(const Eigen::DenseBase<Derived> &input, int32 offset = 0)
    {
        using T = typename Derived::Scalar;
        if (offset == 0) return input.derived().matrix().template triangularView<Eigen::Lower>();
        Eigen::MatrixX<T> result = input.derived().matrix();
        for (Eigen::Index row = 0; row < result.rows(); ++row)
        {
            for (Eigen::Index col = 0; col < result.cols(); ++col)
            {
                if (col - row > offset) result(row, col) = T{};
            }
        }
        return result;
    }

    template <typename T>
    Eigen::MatrixX<T> triu(uint32 rows, uint32 cols, int32 offset = 0)
    {
        return Eigen::MatrixX<T>::NullaryExpr(rows, cols, [offset](Eigen::Index row, Eigen::Index col)
        {
            return static_cast<T>(col - row >= offset);
        });
    }

    template <typename T>
    Eigen::MatrixX<T> triu(uint32 size, int32 offset = 0)
    {
        return triu<T>(size, size, offset);
    }

    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> triu(const Eigen::DenseBase<Derived> &input, int32 offset = 0)
    {
        using T = typename Derived::Scalar;
        if (offset == 0) return input.derived().matrix().template triangularView<Eigen::Upper>();
        Eigen::MatrixX<T> result = input.derived().matrix();
        for (Eigen::Index row = 0; row < result.rows(); ++row)
        {
            for (Eigen::Index col = 0; col < result.cols(); ++col)
            {
                if (col - row < offset) result(row, col) = T{};
            }
        }
        return result;
    }
}
