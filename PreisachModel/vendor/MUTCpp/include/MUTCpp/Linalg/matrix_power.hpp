// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/dot.hpp"
#include "Functions/identity.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// Raise a square matrix to the (integer) power n.
        /// For positive integers n, the power is computed by repeated
        /// matrix squarings and matrix multiplications.  If n == 0,
        /// the identity matrix of the same shape as M is returned.
        /// If n < 0, the inverse is computed and then raised to the abs(n).
        /// @param inArray
        /// @param inPower
        /// @return Matrix
        template <typename dtype>
        Matrix<double> matrix_power(const Matrix<dtype> &inArray, int16 inPower)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const Shape inShape = inArray.shape();
            if (!inShape.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input matrix must be square.");
            }

            if (inPower == 0)
            {
                return identity<double>(inShape.m_Rows);
            }

            if (inPower == 1)
            {
                return inArray.template astype<double>();
            }

            if (inPower == -1)
            {
                return inv(inArray);
            }

            if (inPower > 1)
            {
                Matrix<double> inArrayDouble = inArray.template astype<double>();
                Matrix<double> returnArray = dot(inArrayDouble, inArrayDouble);
                for (int16 i = 2; i < inPower; ++i)
                {
                    returnArray = dot(returnArray, inArrayDouble);
                }
                return returnArray;
            }

            Matrix<double> inverse = inv(inArray);
            Matrix<double> returnArray = dot(inverse, inverse);
            inPower *= -1;
            for (int16 i = 2; i < inPower; ++i)
            {
                returnArray = dot(returnArray, inverse);
            }
            return returnArray;
        }
    }
}
