// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DSArgs.hpp"
#include "DynamicSystem.hpp"

#include <Eigen/Dense>

namespace mc
{
    namespace ode
    {
        inline Ref<ContinuousDS> GetLorenzSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Eigen::Vector3d &x0 = {1.5, -1.5, 20.0})
        {
            auto func = [](Eigen::VectorXd x, double t, const DSArgs &args) -> Eigen::VectorXd
            {
                double sigma = args.at("sigma").toDouble();
                double rho = args.at("rho").toDouble();
                double beta = args.at("beta").toDouble();

                Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
                res[0] = sigma * (x[1] - x[0]);
                res[1] = x[0] * (rho - x[2]) - x[1];
                res[2] = x[0] * x[1] - beta * x[2];
                return res;
            };

            auto jac = [](Eigen::VectorXd x, double t, const DSArgs &args) -> Eigen::MatrixXd
            {
                double sigma = args.at("sigma").toDouble();
                double rho = args.at("rho").toDouble();
                double beta = args.at("beta").toDouble();

                Eigen::MatrixXd res = Eigen::MatrixXd::Zero(x.size(), x.size());
                res(0, 0) = -sigma;
                res(0, 1) = sigma;
                res(1, 0) = rho - x[2];
                res(1, 1) = -1.0;
                res(1, 2) = -x[0];
                res(2, 0) = x[1];
                res(2, 1) = x[0];
                res(2, 2) = -beta;
                return res;
            };

            auto system = Ref<ContinuousDS>::Create(func, jac, dt, args, nextArgs, x0, 0.0);
            return system;
        }
    }
}
