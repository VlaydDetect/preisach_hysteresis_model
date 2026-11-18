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
    auto nanmean(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
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
            return sum / static_cast<double>(count);
        });
    }
}
