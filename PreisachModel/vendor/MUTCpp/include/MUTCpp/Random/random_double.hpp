// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <random>
#include <optional>
#include "Core/Error.hpp"

namespace mc
{
    inline double random_double(double min, double max, std::optional<int> seed = std::nullopt)
    {
        if (min < 0.0 || max <= min) {
            THROW_INVALID_ARGUMENT_ERROR("Interval must be positive and correct: 0 <= min < max");
        }

        static thread_local std::mt19937_64 gen(std::random_device{}());
        if (seed.has_value())
        {
            gen.seed(seed.value());
        }
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
}
