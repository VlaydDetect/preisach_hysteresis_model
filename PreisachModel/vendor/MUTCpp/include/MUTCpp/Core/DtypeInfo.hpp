// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <complex>
#include <limits>

#include "Core/StaticAssert.hpp"

namespace mc {
    /// Holds info about the dtype
    template<typename dtype>
    class DtypeInfo
    {
    public:
        //============================================================================
        /// For integer types: number of non-sign bits in the representation.
        /// For floating types: number of digits(in radix base) in the mantissa
        ///
        /// @return number of bits
        ///
        static constexpr int bits() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::digits;
        }

        //============================================================================
        /// Machine epsilon (the difference between 1 and the least
        /// value greater than 1 that is representable).
        ///
        /// @return dtype
        ///
        static constexpr dtype epsilon() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::epsilon();
        }

        //============================================================================
        /// True if type is integer.
        ///
        /// @return bool
        ///
        static constexpr bool isInteger() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::is_integer;
        }

        //============================================================================
        /// True if type is signed.
        ///
        /// @return bool
        ///
        static constexpr bool isSigned() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::is_signed;
        }

        //============================================================================
        /// Returns the minimum value of the dtype
        ///
        /// @return min value
        ///
        static constexpr dtype min() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::min();
        }

        //============================================================================
        /// Returns the maximum value of the dtype
        ///
        /// @return max value
        ///
        static constexpr dtype max() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::max();
        }
    };

    //================================================================================
    /// Holds info about the std::complex
    template<typename dtype>
    class DtypeInfo<std::complex<dtype>>
    {
    public:
        //============================================================================
        /// For integer types: number of non-sign bits in the representation.
        /// For floating types : number of digits(in radix base) in the mantissa
        ///
        /// @return number of bits
        ///
        static constexpr int bits() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::digits;
        }

        //============================================================================
        /// Machine epsilon (the difference between 1 and the least
        /// value greater than 1 that is representable).
        ///
        /// @return dtype
        ///
        static constexpr std::complex<dtype> epsilon() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return { DtypeInfo<dtype>::epsilon(), DtypeInfo<dtype>::epsilon() };
        }

        //============================================================================
        /// True if type is integer.
        ///
        /// @return bool
        ///
        static constexpr bool isInteger() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::is_integer;
        }

        //============================================================================
        /// True if type is signed.
        ///
        /// @return bool
        ///
        static constexpr bool isSigned() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::numeric_limits<dtype>::is_signed;
        }

        //============================================================================
        /// Returns the minimum value of the dtype
        ///
        /// @return min value
        ///
        static constexpr std::complex<dtype> min() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return { DtypeInfo<dtype>::min(), DtypeInfo<dtype>::min() };
        }

        //============================================================================
        /// Returns the maximum value of the dtype
        ///
        /// @return max value
        ///
        static constexpr std::complex<dtype> max() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return { DtypeInfo<dtype>::max(), DtypeInfo<dtype>::max() };
        }
    };
}