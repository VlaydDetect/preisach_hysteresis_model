// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/argsort.hpp"
#include "Functions/searchsorted.hpp"
#include "Utils/interp.hpp"

namespace mc
{
    template <typename T>
    constexpr double interp(T left, T right, double fraction) noexcept
    {
        return utils::interp(left, right, fraction);
    }

    template <typename Derived, typename Points, typename Values>
    Eigen::RowVectorX<typename Derived::Scalar> interp(const Eigen::DenseBase<Derived> &input,
        const Eigen::DenseBase<Points> &points, const Eigen::DenseBase<Values> &values)
    {
        if (points.size() != values.size() || points.size() == 0)
        {
            throw std::invalid_argument("Interpolation coordinates and values must have the same nonzero size");
        }
        using T = typename Derived::Scalar;
        const auto x = detail::rowMajor(input);
        const auto px = detail::rowMajor(points);
        const auto py = detail::rowMajor(values);
        const Eigen::RowVectorX<Eigen::Index> order = argsort(points).row(0);
        const Eigen::RowVectorXd sorted = px(order).transpose().template cast<double>();
        Eigen::RowVectorX<T> result(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i)
        {
            const double value = static_cast<double>(x(i));
            if (!(value >= sorted(0) && value <= sorted(sorted.size() - 1)))
            {
                throw std::invalid_argument("Interpolation point lies outside the sample range");
            }
            const auto upper = searchsorted(sorted, value);
            if (sorted(upper) == value)
            {
                result(i) = static_cast<T>(py(order(upper)));
            }
            else
            {
                const auto lower = upper - 1;
                const double fraction = (value - sorted(lower)) / (sorted(upper) - sorted(lower));
                result(i) = static_cast<T>(utils::interp(py(order(lower)), py(order(upper)), fraction));
            }
        }
        return result;
    }
}
