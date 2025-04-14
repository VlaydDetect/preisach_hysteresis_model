// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Functions/dot.hpp"
#include "Linalg/inv.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// Solve a linear matrix equation, or system of linear scalar equations.
        /// Computes the “exact” solution, x, of the well-determined, i.e., full rank,
        /// linear matrix equation ax = b.
        /// @param inA
        /// @param inB
        /// @return Matrix<double> Solution to the system a x = b. Returned shape is identical to b
        template <typename dtype>
        Matrix<double> solve(const Matrix<dtype> &inA, const Matrix<dtype> &inB)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (!inA.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array a must be square.");
            }

            if (!inB.isflat())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array b must be flat.");
            }

            if (inA.numCols() != inB.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array b size must be the same as the square size of a.");
            }

            return dot(inv(inA), inB.template astype<double>().reshape(inB.size(), 1)).reshape(inB.shape());
        }
    }
}
