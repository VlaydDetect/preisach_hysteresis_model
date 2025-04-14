// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"
#include "Vector/Vec3.hpp"

namespace mc
{
    namespace linalg
    {
        /// vector hat operator
        /// @return 3x3 Matrix
        template <typename dtype>
        Matrix<dtype> hat(dtype inX, dtype inY, dtype inZ)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            Matrix<dtype> returnArray(3);
            returnArray(0, 0) = 0.;
            returnArray(0, 1) = -inZ;
            returnArray(0, 2) = inY;
            returnArray(1, 0) = inZ;
            returnArray(1, 1) = 0.;
            returnArray(1, 2) = -inX;
            returnArray(2, 0) = -inY;
            returnArray(2, 1) = inX;
            returnArray(2, 2) = 0.;

            return returnArray;
        }

        /// vector hat operator
        /// @param inVec (3x1, or 1x3 cartesian vector)
        /// @return 3x3 Matrix
        template <typename dtype>
        Matrix<dtype> hat(const Matrix<dtype> &inVec)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inVec.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input vector must be a length 3 cartesian vector.");
            }

            return hat(inVec[0], inVec[1], inVec[2]);
        }

        /// vector hat operator
        /// @param inVec
        /// @return 3x3 Matrix
        inline Matrix<double> hat(const Vec3 &inVec)
        {
            return hat(inVec.x, inVec.y, inVec.z);
        }
    }
}
