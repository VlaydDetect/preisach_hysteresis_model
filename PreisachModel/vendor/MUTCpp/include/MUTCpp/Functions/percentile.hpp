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
    namespace detail
    {
        inline double quantileValue(std::vector<double> values, double percent, InterpolationMethod method)
        {
            if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
            std::sort(values.begin(), values.end());
            if (values.size() == 1 || percent == 0.0) return values.front();
            if (percent == 100.0) return values.back();
            const auto position = static_cast<double>(values.size() - 1) * percent / 100.0;
            const auto index = (std::min)(static_cast<std::size_t>(position), values.size() - 2);
            const auto fraction = position - static_cast<double>(index);
            const double lower = values[index];
            const double upper = values[index + 1];
            switch (method)
            {
            case InterpolationMethod::LINEAR: return lower + (upper - lower) * fraction;
            case InterpolationMethod::LOWER: return lower;
            case InterpolationMethod::HIGHER: return upper;
            case InterpolationMethod::NEAREST: return fraction <= 0.5 ? lower : upper;
            case InterpolationMethod::MIDPOINT: return (lower + upper) / 2.0;
            }
            throw std::invalid_argument("Unsupported interpolation method");
        }
    }

    template <typename Derived>
    Eigen::MatrixXd percentile(const Eigen::DenseBase<Derived> &input, double percent, Axis axis = Axis::NONE,
                             InterpolationMethod method = InterpolationMethod::LINEAR)
    {
        if (!(percent >= 0.0 && percent <= 100.0)) throw std::invalid_argument("Percentile must be in [0, 100]");
        if (input.size() == 0) return {};
        return detail::reduce(input, axis, [&](const auto &values)
        {
            return detail::quantileValue(std::vector<double>(values.begin(), values.end()), percent, method);
        });
    }
}
