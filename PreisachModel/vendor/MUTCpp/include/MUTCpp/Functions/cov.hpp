// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <type_traits>

#include "Core/Enums.hpp"
#include "Core/StaticAssert.hpp"
#include "Functions/mean.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Estimate a covariance matrix.
    /// @param x: A 1-D or 2-D array containing multiple variables and observations.
    ///     Each row of x represents a variable, and each column a single observation
    ///     of all those variables.
    /// @param bias: Default normalization (false) is by (N - 1), where N is the number of observations
    ///     given (unbiased estimate). If bias is True, then normalization is by N.
    /// @return Matrix
    template <typename dtype>
    Matrix<double> cov(const Matrix<dtype> &x, Bias bias = Bias::NO)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto varMeans = mean(x, Axis::COL);
        const auto numVars = x.numRows();
        const auto numObs = x.numCols();
        const auto normilizationFactor =
            bias == Bias::YES ? static_cast<double>(numObs) : static_cast<double>(numObs - 1);
        using IndexType = typename std::remove_const<decltype(numVars)>::type;

        // upper triangle
        auto covariance = Matrix<double>(numVars);
        for (IndexType i = 0; i < numVars; ++i)
        {
            const auto var1Mean = varMeans[i];

            for (IndexType j = i; j < numVars; ++j)
            {
                const auto var2Mean = varMeans[j];

                double sum = 0.;
                for (IndexType iObs = 0; iObs < numObs; ++iObs)
                {
                    sum += (x(i, iObs) - var1Mean) * (x(j, iObs) - var2Mean);
                }

                covariance(i, j) = sum / normilizationFactor;
            }
        }

        // fill in the rest of the symmetric matrix
        for (IndexType j = 0; j < numVars; ++j)
        {
            for (IndexType i = j + 1; i < numVars; ++i)
            {
                covariance(i, j) = covariance(j, i);
            }
        }

        return covariance;
    }
}
