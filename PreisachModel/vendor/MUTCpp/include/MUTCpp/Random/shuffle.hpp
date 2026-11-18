// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Random/generator.hpp"

namespace mc::random
{
    template <typename Derived>
    void shuffle(Eigen::DenseBase<Derived> &input)
    {
        auto values = mc::detail::rowMajor(input);
        std::shuffle(values.begin(), values.end(), generator_);
    }
}
