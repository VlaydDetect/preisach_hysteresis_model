// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "copy.hpp"
#include "Matrix.hpp"

namespace mc
{
    template<typename dtype>
    Matrix<dtype> reverse(Matrix<dtype> inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (!inArray.isflat())
        {
            THROW_INVALID_ARGUMENT_ERROR("Matrix must be row (m_Cols = 1) or column (m_Rows = 1)");
        }

        Matrix<dtype> ret = copy(inArray);
        ret.reverse();
        return ret;
    }
}
