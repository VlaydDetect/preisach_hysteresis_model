// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cassert>
#include <numeric>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"
#include "Functions/gcd.hpp"

namespace mc
{
    namespace detail
    {
        template <class T>
        inline constexpr T lcm_imp(const T &a, const T &b) GCD_NOEXCEPT(T)
        {
            T temp = optimal_gcd_select(a, b);
            return temp != T(0) ? T(a / temp * b) : T(0);
        }
    }

    template <typename Integer>
    inline constexpr Integer lcm(Integer const &a, Integer const &b) GCD_NOEXCEPT(Integer)
    {
        return detail::lcm_imp(static_cast<Integer>(detail::gcd_traits<Integer>::abs(a)),
                               static_cast<Integer>(detail::gcd_traits<Integer>::abs(b)));
    }

    /// Returns the least common multiple of |x1| and |x2|.
    /// NOTE: This function uses self-implemented algorithm or a C++17 compiler's lib_gcd_lcm.
    template <typename dtype>
    dtype lcm(dtype inValue1, dtype inValue2) noexcept
    {
        STATIC_ASSERT_INTEGER(dtype);

#if defined(__cpp_lib_gcd_lcm) && 0
        return std::lcm(inValue1, inValue2);
#else
        return lcm(inValue1, inValue2);
#endif
    }

    template <typename I>
    std::pair<typename std::iterator_traits<I>::value_type, I>
    lcm_range(I first, I last) GCD_NOEXCEPT(I)
    {
        assert(first != last);
        typedef typename std::iterator_traits<I>::value_type T;

        T d = *first;
        ++first;
        while (d != T(0) && first != last)
        {
            d = lcm(d, *first);
            ++first;
        }
        return std::make_pair(d, first);
    }

    /// Returns the least common multiple of the values of the input array.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.lcm.html
    ///
    /// @param inArray
    /// @return NdArray<double>
    ///
    template <typename dtype>
    dtype lcm(const Matrix<dtype> &inArray)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return lcm_range(inArray.cbegin(), inArray.cend()).first;
    }
}
