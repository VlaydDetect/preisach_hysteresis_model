// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/take.hpp"

namespace mc
{
    template <typename Derived, typename Indices>
    requires std::is_integral_v<typename Indices::Scalar>
    Eigen::MatrixX<typename Derived::Scalar> deleteIndices(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices, Axis axis = Axis::NONE)
    {
        const auto size = detail::axisSize(input.rows(), input.cols(), axis);
        Eigen::ArrayX<bool> keep = Eigen::ArrayX<bool>::Constant(size, true);
        for (const auto value : detail::rowMajor(indices))
        {
            auto index = static_cast<Eigen::Index>(value);
            if (index < 0) index += size;
            if (index >= 0 && index < size) keep(index) = false;
        }
        Eigen::VectorX<Eigen::Index> positions(keep.count());
        Eigen::Index next = 0;
        for (Eigen::Index i = 0; i < size; ++i)
        {
            if (keep(i)) positions(next++) = i;
        }
        return take(input, positions, axis);
    }

    template <typename Derived>
    Eigen::MatrixX<typename Derived::Scalar> deleteIndices(const Eigen::DenseBase<Derived> &input, Eigen::Index index, Axis axis = Axis::NONE)
    {
        return deleteIndices(input, Eigen::Matrix<Eigen::Index, 1, 1>::Constant(index), axis);
    }

    template <typename Derived, typename First, typename Size, typename Step>
    Eigen::MatrixX<typename Derived::Scalar> deleteIndices(const Eigen::DenseBase<Derived> &input, const Eigen::ArithmeticSequence<First, Size, Step> &indices, Axis axis = Axis::NONE)
    {
        return deleteIndices(input, detail::sequenceIndices(detail::axisSize(input.rows(), input.cols(), axis), indices), axis);
    }
}
