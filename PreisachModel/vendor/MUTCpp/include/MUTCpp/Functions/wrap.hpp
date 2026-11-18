// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/Constants.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include <Eigen/Core>

namespace mc
{
    /**
     * @brief Wrap the input angle to [-pi, pi]
     *
     * @param inAngle: in radians
     * @return Wrapped angle
     */
    template <typename dtype>
        requires std::is_arithmetic_v<dtype>
    double wrap(dtype inAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto angle = std::fmod(static_cast<double>(inAngle) + consts::pi, consts::twoPi);
        if (angle < 0.)
        {
            angle += consts::twoPi;
        }

        return angle - consts::pi;
    }

    /**
     * @brief Wrap the input angle to [-pi, pi]
     *
     * @param inAngles: in radians
     * @return Wrapped angles
     */
    template <typename Derived>
    auto wrap(const Eigen::DenseBase<Derived> &input)
    {
        return input.derived().unaryExpr([&](const auto &value)
        {
            return mc::wrap(value);
        }).eval();
    }
}
