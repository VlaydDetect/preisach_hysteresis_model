// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

namespace mc
{
    /// Enum To describe an axis
    enum class Axis
    {
        NONE = 0,
        ROW,
        COL
    };

    /// Policy for `Matrix` constructor that takes in a pointer to data
    enum class PointerPolicy
    {
        COPY,
        SHELL
    };

    /// Bias boolean
    enum class Bias : bool
    {
        YES = true,
        NO  = false
    };

    /// End Point boolean
    enum class EndPoint : bool
    {
        YES = true,
        NO  = false
    };

    /// Increasing boolean
    enum class Increasing : bool
    {
        YES = true,
        NO  = false
    };

    /// Is Roots boolean
    enum class IsRoots : bool
    {
        YES = true,
        NO  = false
    };

    /// Replace boolean
    enum class Replace : bool
    {
        YES = true,
        NO  = false
    };

    /// Print Results boolean
    enum class PrintResults : bool
    {
        YES = true,
        NO  = false
    };

    /// Right or Left side
    enum class Side
    {
        LEFT,
        RIGHT
    };

    /// Interpolation method
    enum class InterpolationMethod
    {
        LINEAR,
        LOWER,
        HIGHER,
        NEAREST,
        MIDPOINT
    };
}
