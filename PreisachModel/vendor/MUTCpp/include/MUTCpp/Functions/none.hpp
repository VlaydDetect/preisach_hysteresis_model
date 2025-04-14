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
    /// Test whether no array elements along a given axis evaluate to True.
    template<typename dtype>
    Matrix<bool> none(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.none(inAxis);
    }
}
