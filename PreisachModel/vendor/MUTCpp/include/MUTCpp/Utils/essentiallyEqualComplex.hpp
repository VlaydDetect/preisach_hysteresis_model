// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>
#include <string>

#include "Core/DtypeInfo.hpp"
#include "Core/ComplexOperators.hpp"
#include "Core/TypeTraits.hpp"

namespace mc
{
    namespace utils
    {
        /// tests that 2 complex values are "essentially equal"
        template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int>  = 0>
        bool essentiallyEqual(const std::complex<dtype> &inValue1, const std::complex<dtype> &inValue2) noexcept
        {
            return inValue1 == inValue2;
        }

        /// tests that 2 complex values are "essentially equal"
        template <typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int>  = 0>
        bool essentiallyEqual(const std::complex<dtype> &inValue1,
                              const std::complex<dtype> &inValue2,
                              const std::complex<dtype> &inEpsilon) noexcept
        {
            const auto absValue1 = std::abs(inValue1);
            const auto absValue2 = std::abs(inValue2);
            return std::abs(inValue1 - inValue2) <= ((absValue1 > absValue2 ? absValue2 : absValue1) * std::abs(
                inEpsilon));
        }

        /// tests that 2 floating point values are "essentially equal"
        template <typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int>  = 0>
        bool essentiallyEqual(const std::complex<dtype> &inValue1, const std::complex<dtype> &inValue2) noexcept
        {
            return essentiallyEqual(inValue1, inValue2, DtypeInfo<std::complex<dtype>>::epsilon());
        }
    }
}
