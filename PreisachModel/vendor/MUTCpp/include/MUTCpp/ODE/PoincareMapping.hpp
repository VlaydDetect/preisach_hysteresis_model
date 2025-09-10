// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"

namespace mc
{
    namespace ode
    {
        /// Returns a cloud of `numPoints` periodic system solution points
        inline std::vector<Eigen::VectorXd> PoincareMapping(Ref<ode::DynamicalSystem> system, double period, uint32 numPoints)
        {
            if (numPoints == 0)
                return {};

            std::vector<Eigen::VectorXd> points(numPoints);

            for (uint32 i = 0; i < numPoints; i++)
            {
                auto traj = system->Forward(period * i);
                points[i] = traj.row(traj.rows() - 1);
            }

            return points;
        }
    }
}
