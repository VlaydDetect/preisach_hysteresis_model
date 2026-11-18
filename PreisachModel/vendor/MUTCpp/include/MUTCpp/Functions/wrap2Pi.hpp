// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"

#include <cmath>

#include "Core/Constants.hpp"
#include <Eigen/Core>

namespace mc
{
    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @param inAngle: in radians
     * @returns Wrapped angle
     */
    template <typename dtype>
        requires std::is_arithmetic_v<dtype>
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
    template <typename Derived>
    auto wrap2Pi(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::wrap2Pi(value);
        }).eval();
    }
}
