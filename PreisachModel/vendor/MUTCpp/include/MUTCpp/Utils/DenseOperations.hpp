#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Core>

#include "Core/Enums.hpp"

namespace mc::detail
{
    template <typename Derived>
    auto rowMajor(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().template reshaped<Eigen::RowMajor>();
    }

    template <typename Derived>
    auto rowMajor(Eigen::DenseBase<Derived> &input)
    {
        return input.derived().template reshaped<Eigen::RowMajor>();
    }

    inline Eigen::Index axisSize(Eigen::Index rows, Eigen::Index cols, Axis axis)
    {
        switch (axis)
        {
        case Axis::NONE: return rows * cols;
        case Axis::ROW: return rows;
        case Axis::COL: return cols;
        }
        throw std::invalid_argument("Unsupported axis");
    }

    template <typename Derived>
    auto sequenceIndices(Eigen::Index size, const Derived &sequence)
    {
        Eigen::VectorX<Eigen::Index> indices = Eigen::VectorX<Eigen::Index>::LinSpaced(size, 0, size - 1);
        return Eigen::VectorX<Eigen::Index>(indices(sequence));
    }

    inline Eigen::Index checkedIndex(Eigen::Index index, Eigen::Index size, bool allowEnd = false)
    {
        if (index < 0) index += size;
        if (index < 0 || index > size || (!allowEnd && index == size))
        {
            throw std::out_of_range("Index out of range");
        }
        return index;
    }

    template <typename Derived, typename Function>
    auto reduce(const Eigen::DenseBase<Derived> &input, Axis axis, Function function)
    {
        using Result = std::decay_t<decltype(function(rowMajor(input)))>;
        const auto &data = input.derived();
        Eigen::MatrixX<Result> result;
        switch (axis)
        {
        case Axis::NONE:
            result.resize(1, 1);
            result(0, 0) = function(rowMajor(input));
            break;
        case Axis::COL:
            result.resize(1, data.rows());
            for (Eigen::Index row = 0; row < data.rows(); ++row)
            {
                result(0, row) = function(data.row(row));
            }
            break;
        case Axis::ROW:
            result.resize(1, data.cols());
            for (Eigen::Index col = 0; col < data.cols(); ++col)
            {
                result(0, col) = function(data.col(col));
            }
            break;
        default:
            throw std::invalid_argument("Unsupported axis");
        }
        return result;
    }

    template <typename Derived, typename Function>
    Eigen::MatrixX<typename Derived::Scalar> cumulative(
        const Eigen::DenseBase<Derived> &input, Axis axis, Function function)
    {
        using Scalar = typename Derived::Scalar;
        const auto &data = input.derived();
        Eigen::MatrixX<Scalar> result;
        switch (axis)
        {
        case Axis::NONE:
        {
            const auto values = rowMajor(input);
            result.resize(1, values.size());
            std::partial_sum(values.begin(), values.end(), result.row(0).begin(), function);
            break;
        }
        case Axis::COL:
            result.resize(data.rows(), data.cols());
            for (Eigen::Index row = 0; row < data.rows(); ++row)
            {
                const auto values = data.row(row);
                std::partial_sum(values.begin(), values.end(), result.row(row).begin(), function);
            }
            break;
        case Axis::ROW:
            result.resize(data.rows(), data.cols());
            for (Eigen::Index col = 0; col < data.cols(); ++col)
            {
                const auto values = data.col(col);
                std::partial_sum(values.begin(), values.end(), result.col(col).begin(), function);
            }
            break;
        default:
            throw std::invalid_argument("Unsupported axis");
        }
        return result;
    }

    template <typename Derived, typename Function>
    auto mapAxis(const Eigen::DenseBase<Derived> &input, Axis axis, Eigen::Index lengthDelta, Function function)
    {
        using Vector = std::decay_t<decltype(function(rowMajor(input)))>;
        using Scalar = typename Vector::Scalar;
        const auto &data = input.derived();
        if (axis == Axis::NONE)
        {
            return Eigen::MatrixX<Scalar>(function(rowMajor(input)));
        }
        if (axis == Axis::COL)
        {
            Eigen::MatrixX<Scalar> result(data.rows(), (std::max)(Eigen::Index{}, data.cols() + lengthDelta));
            for (Eigen::Index row = 0; row < data.rows(); ++row) result.row(row) = function(data.row(row));
            return result;
        }
        if (axis == Axis::ROW)
        {
            Eigen::MatrixX<Scalar> result((std::max)(Eigen::Index{}, data.rows() + lengthDelta), data.cols());
            for (Eigen::Index col = 0; col < data.cols(); ++col) result.col(col) = function(data.col(col)).transpose();
            return result;
        }
        throw std::invalid_argument("Unsupported axis");
    }

    inline Eigen::Index broadcastSize(Eigen::Index left, Eigen::Index right)
    {
        if (left == right || right == 1) return left;
        if (left == 1) return right;
        throw std::invalid_argument("Incompatible broadcast dimensions");
    }

    template <typename Left, typename Right, typename Function>
    auto broadcast(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right, Function function)
    {
        using Result = std::invoke_result_t<Function, typename Left::Scalar, typename Right::Scalar>;
        Eigen::MatrixX<Result> result(broadcastSize(left.rows(), right.rows()),
                                     broadcastSize(left.cols(), right.cols()));
        for (Eigen::Index row = 0; row < result.rows(); ++row)
        {
            for (Eigen::Index col = 0; col < result.cols(); ++col)
            {
                result(row, col) = function(left(left.rows() == 1 ? 0 : row, left.cols() == 1 ? 0 : col),
                                            right(right.rows() == 1 ? 0 : row, right.cols() == 1 ? 0 : col));
            }
        }
        return result;
    }
}
