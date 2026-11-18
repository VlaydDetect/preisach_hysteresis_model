// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <initializer_list>
#include <vector>

namespace mc
{
    namespace detail
    {
        template <typename Range>
        auto concatenate(const Range &arrays, Axis axis)
        {
            using T = typename Range::value_type::Scalar;
            if (axis != Axis::NONE && axis != Axis::ROW && axis != Axis::COL)
            {
                throw std::invalid_argument("Unsupported axis");
            }

            Eigen::Index length = 0;
            Eigen::Index width = -1;
            for (const auto &array : arrays)
            {
                if (array.rows() == 0 && array.cols() == 0) continue;
                if (axis != Axis::NONE)
                {
                    const auto other = axis == Axis::ROW ? array.cols() : array.rows();
                    if (width < 0) width = other;
                    else if (width != other) throw std::invalid_argument("Incompatible concatenation dimensions");
                }
                length += axisSize(array.rows(), array.cols(), axis);
            }
            width = (std::max)(Eigen::Index{}, width);
            Eigen::MatrixX<T> result(axis == Axis::NONE ? 1 : (axis == Axis::ROW ? length : width),
                                     axis == Axis::ROW ? width : length);
            Eigen::Index next = 0;
            for (const auto &array : arrays)
            {
                if (array.rows() == 0 && array.cols() == 0) continue;
                if (axis == Axis::ROW) result.middleRows(next, array.rows()) = array.matrix();
                else if (axis == Axis::COL) result.middleCols(next, array.cols()) = array.matrix();
                else result.row(0).segment(next, array.size()) = rowMajor(array).transpose().matrix();
                next += axisSize(array.rows(), array.cols(), axis);
            }
            return result;
        }
    }

    template <typename Derived>
    auto concatenate(const std::vector<Derived> &arrays, Axis axis = Axis::NONE)
    {
        return detail::concatenate(arrays, axis);
    }

    template <typename Derived>
    auto concatenate(const std::initializer_list<Derived> &arrays, Axis axis = Axis::NONE)
    {
        return detail::concatenate(arrays, axis);
    }
}
