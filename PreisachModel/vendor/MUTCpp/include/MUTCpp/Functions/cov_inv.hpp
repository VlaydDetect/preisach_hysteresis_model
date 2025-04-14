// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Enums.hpp"
#include "Core/StaticAssert.hpp"
#include "Functions/cov.hpp"
#include "Linalg/inv.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Estimate an inverse covariance matrix, aka the concentration matrix
    /// @param x: A 1-D or 2-D array containing multiple variables and observations.
    ///     Each row of x represents a variable, and each column is a single observation
    ///     of all those variables.
    /// @param bias: Default normalization (false) is by (N - 1), where N is the number of observations
    ///     given (unbiased estimate). If bias is True, then normalization is by N.
    /// @return Matrix
    template<typename dtype>
    Matrix<double> cov_inv(const Matrix<dtype>& x, Bias bias = Bias::NO)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return linalg::inv(cov(x, bias));
    }
}
