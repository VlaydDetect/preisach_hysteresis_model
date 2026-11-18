// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Rotations/Quaternion.hpp"
#include "Functions/round.hpp"
#include <Eigen/LU>

namespace mc::rotations
{
    class DCM
    {
    public:
        static Eigen::Matrix3d eulerAngles(double roll, double pitch, double yaw)
        {
            return Quaternion(roll, pitch, yaw).toDCM();
        }

        static Eigen::Matrix3d eulerAngles(const Eigen::Vector3d &angles)
        {
            return eulerAngles(angles.x(), angles.y(), angles.z());
        }

        static Eigen::Matrix3d eulerAxisAngle(const Eigen::Vector3d &axis, double angle)
        {
            return Quaternion(axis, angle).toDCM();
        }

        template <typename Derived>
        static bool isValid(const Eigen::DenseBase<Derived> &input)
        {
            if (input.rows() != input.cols()) return false;
            const Eigen::MatrixXd matrix = input.derived().matrix().template cast<double>();
            return utils::essentiallyEqual(round(matrix.determinant(), 2), 1.0) &&
                   utils::essentiallyEqual(round(matrix.transpose().determinant(), 2), 1.0);
        }

        static double roll(const Eigen::Matrix3d &dcm) { return Quaternion(dcm).roll(); }
        static double pitch(const Eigen::Matrix3d &dcm) { return Quaternion(dcm).pitch(); }
        static double yaw(const Eigen::Matrix3d &dcm) { return Quaternion(dcm).yaw(); }

        static Eigen::Matrix3d xRotation(double angle) { return Quaternion::xRotation(angle).toDCM(); }
        static Eigen::Matrix3d yRotation(double angle) { return Quaternion::yRotation(angle).toDCM(); }
        static Eigen::Matrix3d zRotation(double angle) { return Quaternion::zRotation(angle).toDCM(); }
    };
}
