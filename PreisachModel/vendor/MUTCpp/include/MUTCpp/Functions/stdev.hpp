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
    auto stdev(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using Result = std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>;
        return detail::reduce(input, axis, [&](const auto &values)
        {
            Result center{};
            for (const auto &value : values) center += static_cast<Result>(value);
            center /= static_cast<double>(values.size());
            Result sum{};
            for (const auto &value : values)
            {
                const auto deviation = static_cast<Result>(value) - center;
                sum += deviation * deviation;
            }
            return std::sqrt(sum / static_cast<double>(values.size()));
        });
    }
}
