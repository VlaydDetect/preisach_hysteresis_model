// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/Constants.hpp"
#include "Matrix.hpp"

namespace mc
{
    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @param inAngle: in radians
     * @returns Wrapped angle
     */
    template <typename dtype>
    double wrap2Pi(dtype inAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto angle = std::fmod(static_cast<double>(inAngle), consts::twoPi);
        if (angle < 0.)
        {
            angle += consts::twoPi;
        }

        return angle;
    }

    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @param inAngles: in radians
     * @returns Wrapped angles
     */
    template <typename dtype>
    Matrix<double> wrap2Pi(const Matrix<dtype> &inAngles) noexcept
    {
        Matrix<double> returnArray(inAngles.size());
        algo::transform(inAngles.begin(),
                        inAngles.end(),
                        returnArray.begin(),
                        [](const auto angle) noexcept -> double { return wrap2Pi(angle); });
        return returnArray;
    }
}
