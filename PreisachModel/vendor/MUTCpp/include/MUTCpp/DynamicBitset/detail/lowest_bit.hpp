// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <assert.h>
#include <climits>
#include <type_traits>
#include <cstdlib>
#include <intrin.h>

namespace mc
{
    namespace detail
    {
        constexpr inline int countl_impl(std::uint32_t x) noexcept
        {
            constexpr unsigned char mod37[37] = {32, 31, 6, 30, 9, 5, 0, 29, 16, 8, 2, 4, 21, 0, 19, 28, 25, 15, 0,
                                                 7, 10, 1, 17, 3, 22, 20, 26, 0, 11, 18, 23, 27, 12, 24, 13, 14, 0};

            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;

            return mod37[x % 37];
        }

        constexpr inline int countl_impl(std::uint8_t x) noexcept
        {
            return countl_impl(static_cast<std::uint32_t>(x)) - 24;
        }

        constexpr inline int countl_impl(std::uint16_t x) noexcept
        {
            return countl_impl(static_cast<std::uint32_t>(x)) - 16;
        }

        constexpr inline int countl_impl(std::uint64_t x) noexcept
        {
            return static_cast<std::uint32_t>(x >> 32) != 0
                ? countl_impl(static_cast<std::uint32_t>(x >> 32))
                : countl_impl(static_cast<std::uint32_t>(x)) + 32;
        }

        template <typename T>
        inline std::enable_if_t<std::is_integral_v<T>, int> integer_log2(T x)
        {
            // We could simply rely on numeric_limits, but sometimes
            // Borland tries to use numeric_limits<const T>, because
            // of its usual const-related problems in argument deduction
            // - gps
            return static_cast<int>((sizeof(T) * CHAR_BIT - 1u) -
                countl_zero(static_cast<std::make_unsigned_t<T>>(x)));
        }

        template <typename T>
        int lowest_bit(T x)
        {

            assert(x >= 1); // PRE

            // clear all bits on except the rightmost one,
            // then calculate logarithm base 2
            //
            return integer_log2<T>(x - (x & (x - 1)));
        }
    }
}
