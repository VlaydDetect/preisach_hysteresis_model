// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Functions/zeros.hpp"
#include "Linalg/svd.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// matrix pseudo-inverse
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
        /// @param inArray
        /// @return Matrix
        template <typename dtype>
        Matrix<double> pinv(const Matrix<dtype> &inArray)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            Matrix<double> u;
            Matrix<double> d;
            Matrix<double> v;
            svd(inArray, u, d, v);

            const auto inShape = inArray.shape();
            auto dPlus = mc::zeros<double>(inShape.cols, inShape.rows); // transpose

            for (uint32 i = 0; i < d.shape().m_Rows; ++i)
            {
                dPlus(i, i) = 1. / d(i, i);
            }

            return v.transpose().dot(dPlus).dot(u.transpose());
        }
    }
}
