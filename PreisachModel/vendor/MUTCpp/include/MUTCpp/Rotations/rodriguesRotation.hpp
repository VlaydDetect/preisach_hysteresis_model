// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Matrix.hpp"
#include "Vector/Vec3.hpp"

namespace mc
{
    namespace rotations
    {
        /// Performs Rodriques' rotation formula
        /// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        /// @param k: the axis to rotate around
        /// @param theta: the angle in radians to rotate
        /// @param v: the vector to rotate
        /// @return Vec3
        inline Vec3 rodriguesRotation(const Vec3 &k, double theta, const Vec3 &v) noexcept
        {
            const auto kUnit = k.normalize();

            const auto vCosTheta = v * std::cos(theta);

            auto kCrossV = kUnit.cross(v);
            kCrossV *= std::sin(theta);

            const auto kDotV = kUnit.dot(v);
            auto kkDotV = kUnit * kDotV;
            kkDotV *= 1 - std::cos(theta);

            auto vec = vCosTheta + kCrossV;
            vec += kkDotV;

            return vec;
        }

        /// Performs Rodriques' rotation formula
        /// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        /// @param k: the axis to rotate around
        /// @param theta: the angle in radians to rotate
        /// @param v: the vector to rotate
        /// @return Matrix<double>
        template <typename dtype>
        Matrix<double> rodriguesRotation(const Matrix<dtype> &k, double theta, const Matrix<dtype> &v)
        {
            return rodriguesRotation(Vec3(k), theta, Vec3(v)).toMatrix();
        }
    }
}
