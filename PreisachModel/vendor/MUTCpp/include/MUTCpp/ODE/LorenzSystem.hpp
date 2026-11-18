// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>

#include "DynamicSystem.hpp"

namespace mc
{
    namespace ode
    {
        struct LorenzParams
        {
            double sigma;
            double rho;
            double beta;
        };

        inline std::unique_ptr<ContinuousDS<3, LorenzParams>> GetLorenzSystem(double dt, const LorenzParams &args, const std::optional<LorenzParams> &nextArgs = {},
                                                 const Eigen::Vector3d &x0 = {1.5, -1.5, 20.0})
        {
            auto func = [](const Eigen::Vector3d &x, double, const LorenzParams &args) -> Eigen::Vector3d
            {
                double sigma = args.sigma;
                double rho = args.rho;
                double beta = args.beta;

                Eigen::Vector3d res;
                res[0] = sigma * (x[1] - x[0]);
                res[1] = x[0] * (rho - x[2]) - x[1];
                res[2] = x[0] * x[1] - beta * x[2];
                return res;
            };

            auto jac = [](const Eigen::Vector3d &x, double, const LorenzParams &args) -> Eigen::Matrix3d
            {
                double sigma = args.sigma;
                double rho = args.rho;
                double beta = args.beta;

                Eigen::Matrix3d res = Eigen::Matrix3d::Zero();
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

            return std::make_unique<ContinuousDS<3, LorenzParams>>(func, jac, dt, args, nextArgs, x0);
        }

        template <typename Derived>
        std::unique_ptr<ContinuousDS<3, LorenzParams>> GetLorenzSystem(double dt, const LorenzParams &args, const std::optional<LorenzParams> &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0)
        {
            return GetLorenzSystem(dt, args, nextArgs, DynamicalSystem<3, LorenzParams>::CheckedState(x0));
        }
    }
}
