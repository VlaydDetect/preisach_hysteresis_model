// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/TypeTraits.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/rms.hpp"
#include "Linalg/inv.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// The Gauss�Newton algorithm is used to solve non-linear least squares problems.
        /// It is a modification of Newton's method for finding a minimum of a function.
        /// https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
        ///
        /// @param numIterations: the number of iterations to perform
        /// @param coordinates: the coordinate values.  The shape needs to be [n x d], where d is
        /// the number of diminsions of the fit function (f(x) is one dimensional,
        /// f(x, y) is two dimensions, etc.), and n is the number of observations
        /// that are being fit to.
        /// @param measurements: the measured values that are being fit
        /// @param function: a std::function of the function that is being fit. The function takes as
        /// inputs a Matrix of a single set of the coordinate values, and a Matrix
        /// of the current values of the fit parameters
        /// @param derivatives: array of std::functions to calculate the function
        /// derivatives. The function that is being fit. The function takes as
        /// inputs a Matrix of a single set of the coordinate values, and a Matrix
        /// of the current values of the fit parameters
        /// @param initialGuess: the initial guess of the parameters to be solved for
        ///
        /// @return pair of Matrix of solved parameter values, and rms of the residuals value
        ///
        template <typename dtype,
                  typename... Params,
                  std::enable_if_t<std::is_arithmetic_v<dtype>, int>  = 0,
                  std::enable_if_t<all_arithmetic_v<Params...>, int>  = 0,
                  std::enable_if_t<all_same_v<dtype, Params...>, int>  = 0>
        std::pair<Matrix<double>, double>
        gaussNewtonNlls(const uint32 numIterations,
                        const Matrix<dtype> &coordinates,
                        const Matrix<dtype> &measurements,
                        const std::function<dtype(const Matrix<dtype> &, const Matrix<dtype> &)> &function,
                        const std::array<std::function<dtype(const Matrix<dtype> &, const Matrix<dtype> &)>,
                                         sizeof...(Params)> &derivatives,
                        Params... initialGuess)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const auto coordinatesShape = coordinates.shape();

            if (coordinatesShape.rows != measurements.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("coordinates number of rows, and measurements size must be the same.");
            }

            Matrix<double> beta = Matrix<dtype>({initialGuess...}).template astype<double>().transpose();
            Matrix<double> residuals(coordinatesShape.rows, 1);
            Matrix<double> jacobian(coordinatesShape.rows, sizeof...(Params));

            const auto colSlice = coordinates.cSlice();
            for (uint32 iteration = 1; iteration <= numIterations; ++iteration)
            {
                for (uint32 measIdx = 0; measIdx < coordinatesShape.rows; ++measIdx)
                {
                    const auto coordinate = coordinates(measIdx, colSlice);

                    residuals[measIdx] =
                        static_cast<double>(measurements[measIdx]) - static_cast<double>(function(coordinate, beta));

                    for (uint32 paramIdx = 0; paramIdx < sizeof...(Params); ++paramIdx)
                    {
                        const auto &derivative = derivatives[paramIdx];
                        jacobian(measIdx, paramIdx) = static_cast<double>(derivative(coordinate, beta));
                    }
                }

                // perform the gauss-newton linear algebra
                const auto jacobianT = jacobian.transpose();
                const auto jacobianPsuedoInverse = linalg::inv(jacobianT.dot(jacobian));
                const auto intermediate = jacobianPsuedoInverse.dot(jacobianT);
                const auto deltaBeta = intermediate.dot(residuals);
                beta += deltaBeta;
            }

            // calculate the final rms of the residuals
            for (uint32 measIdx = 0; measIdx < coordinatesShape.rows; ++measIdx)
            {
                const auto coordinate = coordinates(measIdx, colSlice);

                residuals[measIdx] =
                    static_cast<double>(measurements[measIdx]) - static_cast<double>(function(coordinate, beta));
            }

            return std::make_pair(beta.flatten(), rms(residuals).item());
        }
    }
}
