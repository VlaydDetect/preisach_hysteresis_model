// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"

namespace mc
{
    namespace utils
    {
        /// Raises the input value to an integer power
        template <typename dtype>
        dtype power(dtype inValue, uint8 inPower) noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            if (inPower == 0)
            {
                return static_cast<dtype>(1);
            }

            if (inPower == 1)
            {
                return inValue;
            }

            dtype returnVal = inValue;
            for (uint8 exponent = 1; exponent < inPower; ++exponent)
            {
                returnVal *= inValue;
            }
            return returnVal;
        }

        /// Raises the input value to a floating point power (for example, a fractional degree)
        template <typename dtype1, typename dtype2>
        auto powerf(dtype1 inValue, const dtype2 inPower) noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype1);

            return std::pow(inValue, inPower);
        }

        /// Squares in input value
        template<typename dtype>
        constexpr dtype sqr(dtype inValue) noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            return inValue * inValue;
        }

        /// Cubes in input value
        template<typename dtype>
        constexpr dtype cube(dtype inValue) noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            return inValue * inValue * inValue;
        }
    }
}
