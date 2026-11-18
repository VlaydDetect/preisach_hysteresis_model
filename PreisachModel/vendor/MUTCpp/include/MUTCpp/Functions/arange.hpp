// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mc
{
    template <typename T>
    Eigen::RowVectorX<T> arange(T first, T last, T step = T{1})
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            if (!std::isfinite(first) || !std::isfinite(last) || !std::isfinite(step))
            {
                throw std::invalid_argument("Range endpoints and step must be finite");
            }
        }
        if (step == T{} || (step > T{} && last < first) || (step < T{} && last > first))
        {
            throw std::invalid_argument("Invalid range or step");
        }
        std::vector<T> values;
        for (T value = first; (step > T{} ? value < last : value > last);)
        {
            values.push_back(value);
            T next;
            if constexpr (std::is_integral_v<T>)
            {
                if ((step > T{} && value > (std::numeric_limits<T>::max)() - step) ||
                    (step < T{} && value < (std::numeric_limits<T>::lowest)() - step))
                {
                    break;
                }
                next = static_cast<T>(value + step);
            }
            else
            {
                next = first + step * static_cast<T>(values.size());
            }
            if ((step > T{} && next <= value) || (step < T{} && next >= value))
            {
                throw std::overflow_error("Range no longer advances");
            }
            value = next;
        }
        return Eigen::Map<const Eigen::RowVectorX<T>>(values.data(), static_cast<Eigen::Index>(values.size()));
    }

    template <typename T>
    Eigen::RowVectorX<T> arange(T last)
    {
        if (last <= T{}) throw std::invalid_argument("Range end must be positive");
        return arange(T{}, last);
    }
    template <typename T = Eigen::Index, typename First, typename Size, typename Step>
    Eigen::RowVectorX<T> arange(const Eigen::ArithmeticSequence<First, Size, Step> &indices)
    {
        if (indices.size() < 0) throw std::invalid_argument("Range size cannot be negative");
        return Eigen::RowVectorX<T>::NullaryExpr(indices.size(),
            [&](Eigen::Index i) { return static_cast<T>(indices[i]); });
    }
}
