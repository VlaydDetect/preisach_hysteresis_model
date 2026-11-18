// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>
#include <cmath>

#include "DynamicSystem.hpp"

namespace mc
{
    namespace ode
    {
        inline std::unique_ptr<ContinuousDS<2>> GetRadonsSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Eigen::Vector2d &x0 = {0.0, 0.0}, bool use_atan = false)
        {
            auto func = [use_atan](const Eigen::Vector2d &x, double t, DSArgs &args) -> Eigen::Vector2d
            {
                AL_PROFILE_FUNC("Rodos::func");
                double dt = args.at("dt").toDouble();
                double gamma = args.at("gamma").toDouble();
                double A = args.at("A").toDouble();
                double w0 = args.at("w0").toDouble();
                double w = args.at("w").toDouble();
                double E = args.at("E").toDouble();
                double phi = args.at("phi").toDouble();
                auto &model = args.at("model").toPreisachModel();
                
                Eigen::Vector2d res;

                const double a = A * std::sin(w * t + phi);
                const double g = gamma * x[1];
                const double ww = w0 * x[0];
                const double atan = use_atan ? args.at("eps").toDouble() * std::atan(x[0]) : 0.0;
                const double p = E * model->P(x[0], x[1], static_cast<int>(t / dt));
                const double v = a - g - ww + p + atan;

                res[0] = x[1];
                res[1] = v;

                return res;
            };

            auto jac = [use_atan](const Eigen::Vector2d &x, double t, DSArgs &args) -> Eigen::Matrix2d
            {
                double w0 = args.at("w0").toDouble();
                double gamma = args.at("gamma").toDouble();
                double E = args.at("E").toDouble();
                const auto &model = args.at("model").toPreisachModel();
                double dt = args.at("dt").toDouble();

                Eigen::Matrix2d res = Eigen::Matrix2d::Zero();

                res(0, 1) = 1.0;

                res(1, 0) = -w0 + E * model->DerivativeOperator(t, dt);
                if (use_atan)
                {
                    res(1, 0) += args.at("eps").toDouble() / (1.0 + x[0] * x[0]);
                }
                
                res(1, 1) = -gamma;

                return res;
            };

            return std::make_unique<ContinuousDS<2>>(func, jac, dt, args, nextArgs, x0);
        }

        template <typename Derived>
        std::unique_ptr<ContinuousDS<2>> GetRadonsSystem(double dt, const DSArgs &args, const DSArgs &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0, bool use_atan = false)
        {
            return GetRadonsSystem(dt, args, nextArgs, DynamicalSystem<2>::CheckedState(x0), use_atan);
        }
    }
}
