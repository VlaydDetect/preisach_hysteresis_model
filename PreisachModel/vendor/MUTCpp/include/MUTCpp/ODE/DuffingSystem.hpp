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
        inline std::unique_ptr<ContinuousDS<2>> GetDuffingSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Eigen::Vector2d &x0 = {0.0, 0.0})
        {
            auto func = [](const Eigen::Vector2d &x, double t, const DSArgs &args) -> Eigen::Vector2d
            {
                double delta = args.at("delta").toDouble();
                double alpha = args.at("alpha").toDouble();
                double beta = args.at("beta").toDouble();
                double gamma = args.at("gamma").toDouble();
                double w = args.at("w").toDouble();

                Eigen::Vector2d res;
                res[0] = x[1];
                res[1] = gamma * std::sin(w * t) - delta * x[1] - alpha * x[0] - beta * mc::utils::power(x[0], 3);
                return res;
            };

            auto jac = [](const Eigen::Vector2d &x, double, const DSArgs &args) -> Eigen::Matrix2d
            {
                const double delta = args.at("delta").toDouble();
                const double alpha = args.at("alpha").toDouble();
                const double beta = args.at("beta").toDouble();
                Eigen::Matrix2d res;
                res << 0.0, 1.0,
                       -alpha - 3.0 * beta * x[0] * x[0], -delta;
                return res;
            };

            return std::make_unique<ContinuousDS<2>>(func, jac, dt, args, nextArgs, x0);
        }

        template <typename Derived>
        std::unique_ptr<ContinuousDS<2>> GetDuffingSystem(double dt, const DSArgs &args, const DSArgs &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0)
        {
            return GetDuffingSystem(dt, args, nextArgs, DynamicalSystem<2>::CheckedState(x0));
        }
    }
}
