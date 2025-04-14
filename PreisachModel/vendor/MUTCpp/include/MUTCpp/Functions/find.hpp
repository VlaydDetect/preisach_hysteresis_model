// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <limits>

#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Find flat indices of nonzero elements.
    /// @param mask: the mask to apply to the array
    /// @param n: the first n indices to return (optional, default all)
    /// @return Matrix
    inline Matrix<uint32> find(const Matrix<bool> &mask, uint32 n = std::numeric_limits<uint32>::max())
    {
        Matrix<uint32> indices = mask.flatnonzero();

        if (indices.size() <= n)
        {
            return indices;
        }

        return indices[Slice(0, n)];
    }
}
