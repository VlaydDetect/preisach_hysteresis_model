// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

namespace mc
{
    namespace utils
    {
        //============================================================================
        /// Returns the linear interpolation between two points
        /// @return linear interpolated point
        constexpr double interp(double inValue1, double inValue2, double inPercent) noexcept
        {
            return inValue1 * (1. - inPercent) + inValue2 * inPercent;
        }
    }
}
