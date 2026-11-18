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
    auto nanmedian(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using T = typename Derived::Scalar;
        return detail::reduce(input, axis, [&](const auto &values)
        {
            std::vector<T> filtered;
            for (auto value : values)
            {
                if (!std::isnan(value)) filtered.push_back(value);
            }
            if (filtered.empty()) return std::numeric_limits<T>::quiet_NaN();
            const auto middle = filtered.begin() + filtered.size() / 2;
            std::nth_element(filtered.begin(), middle, filtered.end());
            return *middle;
        });
    }
}
