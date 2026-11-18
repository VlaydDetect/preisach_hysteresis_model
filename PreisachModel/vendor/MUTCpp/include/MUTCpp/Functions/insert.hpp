// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <map>

namespace mc
{
    namespace detail
    {
        template <typename Derived>
        Eigen::MatrixX<typename Derived::Scalar> insertionRows(const Eigen::DenseBase<Derived> &input, Axis axis)
        {
            if (axis == Axis::NONE) return rowMajor(input).matrix();
            if (axis == Axis::ROW) return input.derived().matrix();
            if (axis == Axis::COL) return input.derived().matrix().transpose();
            throw std::invalid_argument("Unsupported axis");
        }
    }

    template <typename Derived, typename Values>
    Eigen::MatrixX<typename Derived::Scalar> insert(const Eigen::DenseBase<Derived> &input, Eigen::Index index, const Eigen::DenseBase<Values> &values, Axis axis = Axis::NONE)
    {
        using T = typename Derived::Scalar;
        const auto rows = detail::insertionRows(input, axis);
        const auto position = detail::checkedIndex(index, rows.rows(), true);

        Eigen::MatrixX<T> inserted;
        if (axis == Axis::NONE) inserted = detail::rowMajor(values).matrix();
        else if (values.size() == 1) inserted = Eigen::MatrixX<T>::Constant(1, rows.cols(), values(0, 0));
        else if (values.size() == rows.cols()) inserted = detail::rowMajor(values).transpose().matrix();
        else inserted = detail::insertionRows(values, axis);
        if (inserted.cols() != rows.cols()) throw std::invalid_argument("Inserted values have incompatible dimensions");

        Eigen::MatrixX<T> result(rows.rows() + inserted.rows(), rows.cols());
        result.topRows(position) = rows.topRows(position);
        result.middleRows(position, inserted.rows()) = inserted;
        result.bottomRows(rows.rows() - position) = rows.bottomRows(rows.rows() - position);
        if (axis != Axis::ROW) return result.transpose();
        return result;
    }

    template <typename Derived, typename Indices, typename Values>
    requires std::is_integral_v<typename Indices::Scalar>
    Eigen::MatrixX<typename Derived::Scalar> insert(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Indices> &indices, const Eigen::DenseBase<Values> &values, Axis axis = Axis::NONE)
    {
        using T = typename Derived::Scalar;
        const auto rows = detail::insertionRows(input, axis);
        const auto positions = detail::rowMajor(indices);
        Eigen::MatrixX<T> inserted;
        if (values.size() == 1) inserted = Eigen::MatrixX<T>::Constant(1, rows.cols(), values(0, 0));
        else if (axis != Axis::NONE && values.size() == rows.cols()) inserted = detail::rowMajor(values).transpose().matrix();
        else inserted = detail::insertionRows(values, axis);
        if (inserted.cols() != rows.cols() || (inserted.rows() != 1 && inserted.rows() != positions.size()))
        {
            throw std::invalid_argument("Inserted values cannot be broadcast to the indices");
        }

        std::map<Eigen::Index, Eigen::Index> insertions;
        for (Eigen::Index i = 0; i < positions.size(); ++i)
        {
            const auto position = detail::checkedIndex(static_cast<Eigen::Index>(positions(i)), rows.rows(), true);
            insertions.try_emplace(position, inserted.rows() == 1 ? 0 : i);
        }
        Eigen::MatrixX<T> result(rows.rows() + static_cast<Eigen::Index>(insertions.size()), rows.cols());
        Eigen::Index start = 0;
        Eigen::Index offset = 0;
        for (const auto &[position, replacement] : insertions)
        {
            result.middleRows(start + offset, position - start) = rows.middleRows(start, position - start);
            result.row(position + offset++) = inserted.row(replacement);
            start = position;
        }
        result.bottomRows(rows.rows() - start) = rows.bottomRows(rows.rows() - start);
        if (axis != Axis::ROW) return result.transpose();
        return result;
    }

    template <typename Derived, typename First, typename Size, typename Step, typename Values>
    Eigen::MatrixX<typename Derived::Scalar> insert(const Eigen::DenseBase<Derived> &input, const Eigen::ArithmeticSequence<First, Size, Step> &indices,
        const Eigen::DenseBase<Values> &values, Axis axis = Axis::NONE)
    {
        const auto positions = detail::sequenceIndices(detail::axisSize(input.rows(), input.cols(), axis), indices);
        return insert(input, positions, values, axis);
    }

    template <typename Derived, typename Indices>
    Eigen::MatrixX<typename Derived::Scalar> insert(const Eigen::DenseBase<Derived> &input, const Indices &indices, typename Derived::Scalar value, Axis axis = Axis::NONE)
    {
        const auto values = Eigen::Matrix<typename Derived::Scalar, 1, 1>::Constant(value);
        return insert(input, indices, values, axis);
    }
}
