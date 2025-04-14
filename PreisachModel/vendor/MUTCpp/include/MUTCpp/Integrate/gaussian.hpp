// @file
// @author Vlayd Detect <gingema62@gmail.com>
// {GitHub Repository}(https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include "Itertools/zip.hpp"

namespace mc
{
    namespace integrate
    {
        namespace detail
        {
            inline std::vector<std::array<double, 2>> dunavant_7_nodes = {
                {1. / 3., 1. / 3.},
                {0.0597159, 0.470142}, {0.470142, 0.0597159}, {0.470142, 0.470142},
                {0.797426, 0.101287}, {0.101287, 0.797426}, {0.101287, 0.101287}
                // {1. / 3., 1. / 3.},
                // {1. / 5., 1. / 5.}, {1. / 5., 3. / 5.}, {3. / 5., 1. / 5.},
                // {1. / 15., 1. / 15.}, {1. / 15., 13. / 15.}, {13. / 15., 1. / 15.}
            };

            inline std::vector dunavant_7_weights =
                {0.225, 0.132394, 0.132394, 0.132394, 0.125939, 0.125939, 0.125939};
                // {9. / 40., 49. / 120., 49. / 120., 49. / 120., 13. / 120., 13. / 120., 13. / 120.};

            inline std::array<double, 2> transformToTriangle(double xi, double eta, Matrix<double> v1,
                                                             Matrix<double> v2,
                                                             Matrix<double> v3)
            {
                auto x = v1[0] * xi + v2[0] * eta + v3[0] * (1 - xi - eta);
                auto y = v1[1] * xi + v2[1] * eta + v3[1] * (1 - xi - eta);
                return {x, y};
            }
        }

        inline double gaussianQuadratureTriangle(const std::function<double(double, double)> &f, Matrix<double> v1,
                                                 Matrix<double> v2, Matrix<double> v3)
        {
            AL_PROFILE_FUNC("mc::integrate::gaussianQuadratureTriangle");
            double area = 0.5 * abs(
                v1[0] * (v2[1] - v3[1]) +
                v2[0] * (v3[1] - v1[1]) +
                v3[0] * (v1[1] - v2[1])
                );

            double integral = 0.0;

            for (const auto &[xi_eta, w] : itertools::zip(detail::dunavant_7_nodes, detail::dunavant_7_weights))
            {
                const auto &[x, y] = detail::transformToTriangle(xi_eta[0], xi_eta[1], v1, v2, v3);
                integral += f(x, y) * w;
            }
            integral *= area;

            return integral;
        }
    }
}
