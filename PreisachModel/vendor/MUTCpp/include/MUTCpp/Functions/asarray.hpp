// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/fromiter.hpp"
#include <initializer_list>
#include <ranges>
#include <stdexcept>

namespace mc
{
    template <std::input_iterator Iterator>
    Eigen::RowVectorX<std::iter_value_t<Iterator>> asarray(Iterator first, Iterator last)
    {
        return fromiter<std::iter_value_t<Iterator>>(first, last);
    }

    template <typename T>
    Eigen::RowVectorX<T> asarray(const T *data, Eigen::Index size)
    {
        if (size < 0 || (size != 0 && !data)) throw std::invalid_argument("Invalid buffer or size");
        return Eigen::Map<const Eigen::RowVectorX<T>>(data, size);
    }

    template <typename T>
    Eigen::MatrixX<T> asarray(const T *data, Eigen::Index rows, Eigen::Index cols)
    {
        if (rows < 0 || cols < 0 || (rows != 0 && cols != 0 && !data))
        {
            throw std::invalid_argument("Invalid buffer or dimensions");
        }
        using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        return Eigen::Map<const RowMajorMatrix>(data, rows, cols);
    }

    template <typename Range>
    requires (std::ranges::common_range<const Range> && !std::ranges::range<std::ranges::range_value_t<Range>>)
    auto asarray(const Range &range)
    {
        return asarray(std::ranges::begin(range), std::ranges::end(range));
    }

    template <std::ranges::sized_range Range>
    requires std::ranges::forward_range<std::ranges::range_value_t<Range>>
    auto asarray(const Range &range)
    {
        using T = std::ranges::range_value_t<std::ranges::range_value_t<Range>>;
        const auto rows = static_cast<Eigen::Index>(std::ranges::size(range));
        const auto cols = rows == 0 ? 0 : static_cast<Eigen::Index>(std::ranges::distance(*std::ranges::begin(range)));
        Eigen::MatrixX<T> result(rows, cols);
        Eigen::Index row = 0;
        for (const auto &values : range)
        {
            if (std::ranges::distance(values) != cols) throw std::invalid_argument("Rows must have the same length");
            std::ranges::copy(values, result.row(row++).begin());
        }
        return result;
    }

    template <typename T>
    auto asarray(std::initializer_list<T> values)
    {
        return asarray(values.begin(), values.end());
    }

    template <typename T>
    auto asarray(std::initializer_list<std::initializer_list<T>> values)
    {
        return asarray<decltype(values)>(values);
    }
}
