// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Shape.hpp"
#include "Core/Slice.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Pads an array.
    template <typename dtype>
    Matrix<dtype> pad(const Matrix<dtype> &inArray, uint16 inPadWidth, dtype inPadValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();
        Shape outShape(inShape);
        outShape.m_Rows += 2 * inPadWidth;
        outShape.m_Cols += 2 * inPadWidth;

        Matrix<dtype> returnArray(outShape);
        returnArray.fill(inPadValue);
        returnArray.put(Slice(inPadWidth, inPadWidth + inShape.m_Rows),
                        Slice(inPadWidth, inPadWidth + inShape.m_Cols),
                        inArray);

        return returnArray;
    }
}
