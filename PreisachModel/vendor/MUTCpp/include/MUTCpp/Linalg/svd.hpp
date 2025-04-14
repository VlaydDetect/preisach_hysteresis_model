// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <utility>

#include "Core/StaticAssert.hpp"
#include "Functions/diagflat.hpp"
#include "Linalg/svd/SVDClass.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// matrix svd
    /// @param inArray: Matrix to be SVDed
    /// @param outU: Matrix output U
    /// @param outS: Matrix output S
    /// @param outVt: Matrix output V transpose
        template <typename dtype>
        void svd(const Matrix<dtype> &inArray, Matrix<double> &outU, Matrix<double> &outS, Matrix<double> &outVt)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            SVD svdSolver(inArray.template astype<double>());
            outU = svdSolver.u();

            Matrix<double> vt = svdSolver.v().transpose();
            outVt = std::move(vt);

            Matrix<double> s = diagflat(svdSolver.s(), 0);
            outS = std::move(s);
        }
    }
}
