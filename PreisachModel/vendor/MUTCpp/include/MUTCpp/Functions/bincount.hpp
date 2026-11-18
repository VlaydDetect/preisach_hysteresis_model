// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <limits>
#include <utility>

namespace mc
{
    namespace detail
    {
        template <typename Derived, typename Weight>
        Eigen::RowVectorX<typename Derived::Scalar> bincount(
            const Eigen::DenseBase<Derived> &input, Eigen::Index minimumLength, Weight weight)
        {
            using T = typename Derived::Scalar;
            static_assert(std::is_integral_v<T>);
            if (minimumLength < 0) throw std::invalid_argument("Minimum length must be nonnegative");
            if (input.size() == 0) return Eigen::RowVectorX<T>::Zero(minimumLength);
            const T maximum = input.maxCoeff();
            if (maximum < T{}) return Eigen::RowVectorX<T>(0);
            if (maximum == (std::numeric_limits<T>::max)() ||
                std::cmp_greater_equal(maximum, (std::numeric_limits<Eigen::Index>::max)()))
            {
                throw std::invalid_argument("Bin index is too large");
            }

            const auto length = (std::max)(minimumLength, static_cast<Eigen::Index>(maximum) + 1);
            Eigen::RowVectorX<T> counts = Eigen::RowVectorX<T>::Zero(length);
            const auto values = rowMajor(input);
            for (Eigen::Index i = 0; i < values.size(); ++i)
            {
                counts(static_cast<Eigen::Index>((std::max)(T{}, values(i)))) += weight(i);
            }
            return counts;
        }
    }

    template <typename Derived>
    auto bincount(const Eigen::DenseBase<Derived> &input, Eigen::Index minimumLength = 1)
    {
        return detail::bincount(input, minimumLength, [](Eigen::Index) { return 1; });
    }

    template <typename Derived, typename Weights>
    auto bincount(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Weights> &weights, Eigen::Index minimumLength = 1)
    {
        if (input.rows() != weights.rows() || input.cols() != weights.cols())
        {
            throw std::invalid_argument("Input and weights must have the same shape");
        }
        const auto values = detail::rowMajor(weights);
        return detail::bincount(input, minimumLength, [&](Eigen::Index i) { return values(i); });
    }
}
