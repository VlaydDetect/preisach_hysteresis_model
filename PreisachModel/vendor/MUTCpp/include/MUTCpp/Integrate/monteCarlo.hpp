// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <array>

#include "Random/rand.hpp"

namespace mc
{
    namespace integrate
    {
        namespace detail
        {
            inline std::array<Matrix<double>, 2> sampleTriangle(Matrix<double> v1, Matrix<double> v2, Matrix<double> v3,
                                                                uint32 n)
            {
                const Matrix<double> u = mc::random::rand<double>(Shape(1, n));
                const Matrix<double> v = mc::random::rand<double>(Shape(1, n));
                const auto u_sqrt = sqrt(u);

                const auto x = v1[0] * (1. - u_sqrt) + v2[0] * (u_sqrt * (1. - v)) + v3[0] * (u_sqrt * v);
                const auto y = v1[1] * (1. - u_sqrt) + v2[1] * (u_sqrt * (1. - v)) + v3[1] * (u_sqrt * v);

                return {x, y};
            }
        }

        inline double monteCarloTriangle(std::function<double(double, double)> f, Matrix<double> v1, Matrix<double> v2,
                                         Matrix<double> v3, uint32 numSamples = 5000, double tol = 1e-3,
                                         uint32 maxIter = 4)
        {
            AL_PROFILE_FUNC("mc::integrate::monteCarloTriangle");
            double area = 0.5 * abs(
                v1[0] * (v2[1] - v3[1]) +
                v2[0] * (v3[1] - v1[1]) +
                v3[0] * (v1[1] - v2[1])
                );

            double prev_integral = consts::nan;
            for (uint32 i = 0; i < maxIter; ++i)
            {
                const auto &[x, y] = detail::sampleTriangle(v1, v2, v3, numSamples);
                Matrix<double> integral(1, numSamples);
                for (uint32 j = 0; j < numSamples; ++j)
                {
                    integral[j] = f(x[j], y[j]);
                }
                integral = area * mean(integral);

                auto div = abs(integral - prev_integral)[0];

                if (!isnan(prev_integral) && div < tol)
                {
                    return integral[0];
                }

                prev_integral = integral[0];
                numSamples *= 2;
            }

            return prev_integral;
        }
    }
}
