// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Geometry>

namespace mc::rotations
{
    inline Eigen::Vector3d rodriguesRotation(const Eigen::Vector3d &axis, double angle, const Eigen::Vector3d &vector) noexcept
    {
        return Eigen::AngleAxisd(angle, axis / axis.norm()) * vector;
    }
}
