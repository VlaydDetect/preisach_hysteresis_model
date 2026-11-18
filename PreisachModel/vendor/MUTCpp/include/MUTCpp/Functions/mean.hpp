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
    auto mean(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using Result = std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>;
        return detail::reduce(input, axis, [&](const auto &values)
        {
            Result sum{};
            for (const auto &value : values) sum += static_cast<Result>(value);
            return sum / static_cast<double>(values.size());
        });
    }
}
