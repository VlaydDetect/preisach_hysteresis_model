// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Linalg/svd/SVDClass.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// Solves the equation Ax = b by computing a vector x
    /// that minimizes the Euclidean 2-norm || b - Ax ||^2.
    /// The equation may be under-, well-, or over-determined
    /// (i.e., the number of linearly independent rows of a can
    /// be less than, equal to, or greater than its number of
    /// linearly independent columns). If A is square and of
    /// full rank, then x (but for round-off error) is the
    /// "exact" solution of the equation.
    /// @param inA: coefficient matrix
    /// @param inB: Ordinate or "dependent variable" values
    /// @param inTolerance (default 1e-12)
    /// @return Matrix
        template <typename dtype>
        Matrix<double> lstsq(const Matrix<dtype> &inA, const Matrix<dtype> &inB, double inTolerance = 1e-12)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            SVD svdSolver(inA.template astype<double>());
            const double threshold = inTolerance * svdSolver.s().front();

            return svdSolver.solve(inB.template astype<double>(), threshold);
        }
    }
}
