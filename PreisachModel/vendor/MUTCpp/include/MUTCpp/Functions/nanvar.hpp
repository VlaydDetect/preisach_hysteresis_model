// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Functions/nanstdev.hpp"
#include "Functions/square.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the variance along the specified axis, while ignoring NaNs.
    template<typename dtype>
    Matrix<double> nanvar(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        return square(nanstdev(inArray, inAxis));
    }
}
