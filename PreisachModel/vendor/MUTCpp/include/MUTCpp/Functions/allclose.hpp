// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Functions/abs.hpp"
#include "Functions/all.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns True if two arrays are element-wise equal within a tolerance.
    /// inTolerance must be a positive number
    template<typename dtype>
    bool allclose(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2, double inTolerance = 1e-5)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array dimensions are not consistant.");
        }

        for (uint32 i = 0; i < inArray1.size(); ++i)
        {
            if (std::abs(inArray1[i] - inArray2[i]) > inTolerance)
            {
                return false;
            }
        }

        return true;
    }
}
