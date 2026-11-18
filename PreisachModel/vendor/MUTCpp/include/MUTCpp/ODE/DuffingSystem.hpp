// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>
#include <cmath>

#include "DynamicSystem.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    namespace ode
    {
        struct DuffingParams
        {
            double delta;
            double alpha;
            double beta;
            double gamma;
            double w;
        };

        inline std::unique_ptr<ContinuousDS<2, DuffingParams>> GetDuffingSystem(double dt, const DuffingParams &args, const std::optional<DuffingParams> &nextArgs = {},
                                                 const Eigen::Vector2d &x0 = {0.0, 0.0})
        {
            auto func = [](const Eigen::Vector2d &x, double t, const DuffingParams &args) -> Eigen::Vector2d
            {
                double delta = args.delta;
                double alpha = args.alpha;
                double beta = args.beta;
                double gamma = args.gamma;
                double w = args.w;

                Eigen::Vector2d res;
                res[0] = x[1];
                res[1] = gamma * std::sin(w * t) - delta * x[1] - alpha * x[0] - beta * mc::utils::power(x[0], 3);
                return res;
            };

            auto jac = [](const Eigen::Vector2d &x, double, const DuffingParams &args) -> Eigen::Matrix2d
            {
                const double delta = args.delta;
                const double alpha = args.alpha;
                const double beta = args.beta;
                Eigen::Matrix2d res;
                res << 0.0, 1.0,
                       -alpha - 3.0 * beta * x[0] * x[0], -delta;
                return res;
            };

            return std::make_unique<ContinuousDS<2, DuffingParams>>(func, jac, dt, args, nextArgs, x0);
        }

        template <typename Derived>
        std::unique_ptr<ContinuousDS<2, DuffingParams>> GetDuffingSystem(double dt, const DuffingParams &args, const std::optional<DuffingParams> &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0)
        {
            return GetDuffingSystem(dt, args, nextArgs, DynamicalSystem<2, DuffingParams>::CheckedState(x0));
        }
    }
}
