// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/percentile.hpp"

namespace mc
{
    template <typename Derived>
    Eigen::MatrixXd nanpercentile(const Eigen::DenseBase<Derived> &input, double percent, Axis axis = Axis::NONE,
                                InterpolationMethod method = InterpolationMethod::LINEAR)
    {
        if (!(percent >= 0.0 && percent <= 100.0)) throw std::invalid_argument("Percentile must be in [0, 100]");
        if (input.size() == 0) return {};
        return detail::reduce(input, axis, [&](const auto &values)
        {
            std::vector<double> filtered;
            for (double value : values)
            {
                if (!std::isnan(value)) filtered.push_back(value);
            }
            return detail::quantileValue(std::move(filtered), percent, method);
        });
    }
}
