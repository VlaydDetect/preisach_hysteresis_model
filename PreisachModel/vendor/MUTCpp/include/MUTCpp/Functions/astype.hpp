// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"

namespace mc
{
    /// Returns a copy of the array, cast to a specified type.
    template<typename dtypeOut = double, typename dtype>
    Matrix<dtypeOut> astype(const Matrix<dtype> inArray)
    {
        return inArray.template astype<dtypeOut>();
    }
}
