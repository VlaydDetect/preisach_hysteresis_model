// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Matrix/MatrixCore.hpp"

namespace mc
{
    /// Interpret a buffer as a 1-dimensional array.
    template <typename dtype>
    Matrix<dtype> frombuffer(const char *inBufferPtr, uint32 inNumBytes)
    {
        if (inNumBytes % sizeof(dtype) != 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("inNumBytes % sizeof(dtype) != 0");
        }

        const auto numElements = static_cast<uint32>(inNumBytes / sizeof(dtype));
        return NdArray<dtype>(reinterpret_cast<const dtype *>(inBufferPtr), numElements);
    }
}
