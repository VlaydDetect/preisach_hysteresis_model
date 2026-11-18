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
    auto nanstdev(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        return detail::reduce(input, axis, [&](const auto &values)
        {
            double sum = 0.0;
            Eigen::Index count = 0;
            for (double value : values)
            {
                if (!std::isnan(value))
                {
                    sum += value;
                    ++count;
                }
            }
            const double mean = sum / static_cast<double>(count);
            sum = 0.0;
            for (double value : values)
            {
                if (!std::isnan(value)) sum += (value - mean) * (value - mean);
            }
            return std::sqrt(sum / static_cast<double>(count));
        });
    }
}
