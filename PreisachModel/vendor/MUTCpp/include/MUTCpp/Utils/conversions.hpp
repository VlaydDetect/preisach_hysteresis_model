// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <vector>

#include "Core/Error.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Converts two vectors into a vector of pairs of elements with the same indexes
    template <typename T, typename U>
    std::vector<std::pair<T, U>> vecs2pairs(std::vector<T> a, std::vector<U> b)
    {
        if (a.size() != b.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("The vectors must have the same size!");
        }

        std::vector<std::pair<T, U>> result(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            result[i] = std::make_pair(a[i], b[i]);
        }

        return result;
    }
}
