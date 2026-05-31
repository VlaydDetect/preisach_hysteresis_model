// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DSArgs.hpp"
#include "DynamicSystem.hpp"

#include <Eigen/Dense>

#include "Functions/power.hpp"

namespace mc
{
    namespace ode
    {
        inline Ref<ContinuousDS> GetDuffingSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Eigen::Vector2d &x0 = {0.0, 0.0})
        {
            auto func = [](Eigen::VectorXd x, double t, const DSArgs &args) -> Eigen::VectorXd
            {
                double delta = args.at("delta").toDouble();
                double alpha = args.at("alpha").toDouble();
                double beta = args.at("beta").toDouble();
                double gamma = args.at("gamma").toDouble();
                double w = args.at("w").toDouble();

                Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
                res[0] = x[1];
                res[1] = gamma * sin(w * t) - delta * x[1] - alpha * x[0] - beta * mc::power(x[0], 3);
                return res;
            };

            auto jac = [](Eigen::VectorXd x, double t, const DSArgs &args) -> Eigen::MatrixXd
            {
                Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
                return res;
            };

            auto system = Ref<ContinuousDS>::Create(func, jac, dt, args, nextArgs, x0, 0.0);
            return system;
        }
    }
}
