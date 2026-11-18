// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>
#include <cmath>

#include "DynamicSystem.hpp"
#include "Hysteresis/ModelBase.hpp"

namespace mc
{
    namespace ode
    {
        struct RadonsParams
        {
            double dt;
            double gamma;
            double A;
            double w0;
            double w;
            double E;
            double phi = 0.0;
            double eps = 0.0;
        };

        struct RadonsContext
        {
            RadonsParams params;
            std::shared_ptr<PreisachModelBase> model;

            static void ValidatePair(const RadonsContext &primary, const std::optional<RadonsContext> &perturbed)
            {
                if (!primary.model || (perturbed && !perturbed->model))
                {
                    throw std::invalid_argument("Each Radons trajectory requires a Preisach model");
                }
                if (perturbed && primary.model == perturbed->model)
                {
                    throw std::invalid_argument("Radons trajectories require distinct Preisach models");
                }
            }
        };

        using RadonsSystem = ContinuousDS<2, RadonsContext>;

        /// Omit nextArgs for a single trajectory. A paired calculation requires
        /// a second context with a separately constructed Preisach model.
        inline std::unique_ptr<RadonsSystem> GetRadonsSystem(double dt, const RadonsContext &args,
            const std::optional<RadonsContext> &nextArgs = {}, const Eigen::Vector2d &x0 = {0.0, 0.0},
            bool use_atan = false)
        {
            auto func = [use_atan](const Eigen::Vector2d &x, double t, const RadonsContext &args) -> Eigen::Vector2d
            {
                AL_PROFILE_FUNC("Rodos::func");
                const auto &[dt, gamma, A, w0, w, E, phi, eps] = args.params;
                
                Eigen::Vector2d res;

                const double a = A * std::sin(w * t + phi);
                const double g = gamma * x[1];
                const double ww = w0 * x[0];
                const double atan = use_atan ? eps * std::atan(x[0]) : 0.0;
                const double p = E * args.model->P(x[0], x[1], static_cast<int>(t / dt));
                const double v = a - g - ww + p + atan;

                res[0] = x[1];
                res[1] = v;

                return res;
            };

            auto jac = [use_atan](const Eigen::Vector2d &x, double t, const RadonsContext &args) -> Eigen::Matrix2d
            {
                const auto &params = args.params;

                Eigen::Matrix2d res = Eigen::Matrix2d::Zero();

                res(0, 1) = 1.0;

                res(1, 0) = -params.w0 + params.E * args.model->DerivativeOperator(t, params.dt);
                if (use_atan)
                {
                    res(1, 0) += params.eps / (1.0 + x[0] * x[0]);
                }
                
                res(1, 1) = -params.gamma;

                return res;
            };

            return std::make_unique<RadonsSystem>(func, jac, dt, args, nextArgs, x0);
        }

        template <typename Derived>
        std::unique_ptr<RadonsSystem> GetRadonsSystem(double dt, const RadonsContext &args,
                                                         const std::optional<RadonsContext> &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0, bool use_atan = false)
        {
            return GetRadonsSystem(dt, args, nextArgs, RadonsSystem::CheckedState(x0), use_atan);
        }
    }
}
