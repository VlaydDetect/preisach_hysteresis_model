// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the length of the first dimension of the input array.
    /// @return length uint16
    template <typename dtype>
    uint32 alen(const Matrix<dtype> &inArray) noexcept
    {
        return inArray.shape().rows;
    }
}
