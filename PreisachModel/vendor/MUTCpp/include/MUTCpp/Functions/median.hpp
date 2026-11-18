// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include "Core/TypeTraits.hpp"

namespace mc
{
    template <typename Derived>
    auto median(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using T = typename Derived::Scalar;
        return detail::reduce(input, axis, [&](const auto &values)
        {
            std::vector<T> sorted(values.begin(), values.end());
            if (sorted.empty()) throw std::invalid_argument("Empty median");
            const auto mid = sorted.size() / 2;
            std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
            if (sorted.size() % 2 != 0) return sorted[mid];
            const auto low = std::max_element(sorted.begin(), sorted.begin() + mid);
            return static_cast<T>((*low + sorted[mid]) / T{2});
        });
    }
}
