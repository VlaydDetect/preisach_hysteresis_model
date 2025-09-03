// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"
#include "DSArgs.hpp"
#include "DynamicSystem.hpp"
#include "Functions.hpp"

namespace mc
{
    namespace ode
    {
        inline Ref<ContinuousDS> GetLorenzSystem(double dt, const DSArgs &args, const DSArgs &nextArgs = {},
                                                 const Matrix<double> &x0 = {1.5, -1.5, 20.0})
        {
            auto func = [](Matrix<double> x, double t, const DSArgs &args) -> Matrix<double>
            {
                double sigma = args.at("sigma").toDouble();
                double rho = args.at("rho").toDouble();
                double beta = args.at("beta").toDouble();

                auto res = mc::zeros_like<double>(x);
                res[0] = sigma * (x[1] - x[0]);
                res[1] = x[0] * (rho - x[2]) - x[1];
                res[2] = x[0] * x[1] - beta * x[2];
                return res;
            };

            auto jac = [](Matrix<double> x, double t, const DSArgs &args) -> Matrix<double>
            {
                double sigma = args.at("sigma").toDouble();
                double rho = args.at("rho").toDouble();
                double beta = args.at("beta").toDouble();

                auto res = mc::zeros<double>(x.shape().m_Cols);
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
