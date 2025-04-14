// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Functions/cov.hpp"
#include "Functions/empty_like.hpp"
#include "Functions/sqrt.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return Pearson product-moment correlation coefficients.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    ///
    /// @param x: A 1-D or 2-D array containing multiple variables and observations.
    /// Each row of x represents a variable, and each column is a single observation
    /// of all those variables.
    /// @return Matrix
    ///
    template <typename dtype>
    Matrix<double> corrcoef(const Matrix<dtype> &x)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto covariance = cov(x);
        auto normalizedCovariance = empty_like(covariance);
        for (decltype(covariance.numRows()) i = 0; i < covariance.numRows(); ++i)
        {
            for (decltype(covariance.numCols()) j = 0; j < covariance.numCols(); ++j)
            {
                normalizedCovariance(i, j) = covariance(i, j) / sqrt(covariance(i, i) * covariance(j, j));
            }
        }

        return normalizedCovariance;
    }
}
