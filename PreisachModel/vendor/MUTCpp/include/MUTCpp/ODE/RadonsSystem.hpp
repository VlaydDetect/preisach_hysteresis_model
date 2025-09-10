// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DSArgs.hpp"
#include "DynamicSystem.hpp"

namespace mc
{
    namespace ode
    {
        inline Ref<ContinuousDS> GetRadonsSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Eigen::Vector2d &x0 = {0.0, 0.0})
        {
            DynamicalSystem::DSFunc func = [](const Eigen::VectorXd &x, const double t,
                                              DSArgs &args) -> Eigen::VectorXd
            {
                AL_PROFILE_FUNC("Rodos::func");
                double dt = args.at("dt").toDouble();
                double gamma = args.at("gamma").toDouble();
                double A = args.at("A").toDouble();
                double w0 = args.at("w0").toDouble();
                double w = args.at("w").toDouble();
                double E = args.at("E").toDouble();
                auto &model = args.at("model").toPreisachModel();

                Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());

                res[0] = x[1];
                res[1] = A * cos(w * t) - gamma * x[1] - power(w0, 2) * x[0] + E * model->P(
                    x[0], static_cast<int>(t / dt));

                return res;
            };

            auto jac = [](Eigen::VectorXd x, double t, DSArgs &args) -> Eigen::MatrixXd
            {
                double w0 = args.at("w0").toDouble();
                double gamma = args.at("gamma").toDouble();
                double E = args.at("E").toDouble();
                auto model = args.at("model").toPreisachModel();
                double dt = args.at("dt").toDouble();
                double A = args.at("A").toDouble();

                Eigen::MatrixXd res = Eigen::MatrixXd::Zero(x.size(), x.size());

                res(0, 1) = 1.0;

                res(1, 0) = -w0 + E * model->DerivativeOperator(t, dt);
                res(1, 1) = -gamma;
                res(1, 2) = A * cos(x[2]);

                return res;
            };

            auto system = Ref<ContinuousDS>::Create(func, jac, dt, args, nextArgs, x0);

            return system;
        }
    }
}
