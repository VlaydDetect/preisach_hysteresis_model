// @file
// @author Vlayd Detect <gingema62@gmail.com>
// {GitHub Repository}(https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <array>
#include <cmath>
#include <functional>

#include <Eigen/Core>

#include "Utils/Profile.hpp"

namespace mc
{
    namespace integrate
    {
        namespace detail
        {
            inline constexpr std::array<std::array<double, 2>, 7> dunavant_7_nodes = {{
                {1. / 3., 1. / 3.},
                {0.0597159, 0.470142}, {0.470142, 0.0597159}, {0.470142, 0.470142},
                {0.797426, 0.101287}, {0.101287, 0.797426}, {0.101287, 0.101287}
                // {1. / 3., 1. / 3.},
                // {1. / 5., 1. / 5.}, {1. / 5., 3. / 5.}, {3. / 5., 1. / 5.},
                // {1. / 15., 1. / 15.}, {1. / 15., 13. / 15.}, {13. / 15., 1. / 15.}
            }};

            inline constexpr std::array<double, 7> dunavant_7_weights =
                {0.225, 0.132394, 0.132394, 0.132394, 0.125939, 0.125939, 0.125939};
                // {9. / 40., 49. / 120., 49. / 120., 49. / 120., 13. / 120., 13. / 120., 13. / 120.};

            inline Eigen::Vector2d transformToTriangle(double xi, double eta, const Eigen::Vector2d &v1,
                                                       const Eigen::Vector2d &v2, const Eigen::Vector2d &v3)
            {
                return xi * v1 + eta * v2 + (1.0 - xi - eta) * v3;
            }
        }

        inline double gaussianQuadratureTriangle(const std::function<double(double, double)> &f,
                                                 const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                 const Eigen::Vector2d &v3)
        {
            AL_PROFILE_FUNC("mc::integrate::gaussianQuadratureTriangle");
            double area = 0.5 * std::abs(
                v1[0] * (v2[1] - v3[1]) +
                v2[0] * (v3[1] - v1[1]) +
                v3[0] * (v1[1] - v2[1])
                );

            double integral = 0.0;

            for (std::size_t i = 0; i < detail::dunavant_7_nodes.size(); ++i)
            {
                const auto &[xi, eta] = detail::dunavant_7_nodes[i];
                const auto point = detail::transformToTriangle(xi, eta, v1, v2, v3);
                integral += f(point.x(), point.y()) * detail::dunavant_7_weights[i];
            }
            integral *= area;

            return integral;
        }
    }
}
