// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

# include <type_traits>
# include <bit>

#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
#define GCD_NOEXCEPT(T) noexcept(std::is_arithmetic<T>::value)

    namespace type_traits
    {
        template <bool B, class T = void>
        struct enable_if_c
        {
            typedef T type;
        };

        template <class T>
        struct enable_if_c<false, T>
        {
        };
    }

    namespace detail
    {
        // some helper functions that really should be constexpr already, but sadly aren't:
        template <class T>
        inline constexpr T constexpr_min(T const &a, T const &b) GCD_NOEXCEPT(T)
        {
            return a < b ? a : b;
        }

        template <class T>
        inline constexpr auto constexpr_swap(T &a, T &b) GCD_NOEXCEPT(T) -> decltype(a.swap(b))
        {
            return a.swap(b);
        }

        template <class T, class U>
        inline constexpr void constexpr_swap(T &a, U &b...) GCD_NOEXCEPT(T)
        {
            T t(static_cast<T &&>(a));
            a = static_cast<T &&>(b);
            b = static_cast<T &&>(t);
        }

        // Like std::abs, but supports unsigned types and returns the specified type,
        // so |std::numeric_limits<_Tp>::min()| is OK if representable in Res.
        template <typename Res, typename Tp>
        constexpr Res abs_r(Tp val)
        {
            static_assert(sizeof(Res) >= sizeof(Tp), "result type must be at least as wide as the input type");

            if (val >= 0)
            {
                return val;
            }

            return -static_cast<Res>(val);
        }

        /// GCD implementation, using Stein's algorithm
        template <typename dtype>
        constexpr dtype gcd(dtype m, dtype n)
        {
            STATIC_ASSERT_UNSIGNED_INTEGER(dtype);

            if (m == 0)
            {
                return n;
            }
            if (n == 0)
            {
                return m;
            }

            const int i = std::countr_zero<dtype>(m);
            m >>= i;
            const int j = std::countl_zero<dtype>(n);
            n >>= j;
            const int k = i < j ? i : j;

            while (true)
            {
                if (m > n)
                {
                    dtype tmp = m;
                    m = n;
                    n = tmp;
                }

                n -= m;

                if (n == 0)
                    return m << k;

                n >>= std::countr_zero<dtype>(n);
            }
        }

        template <class T, bool a = std::is_unsigned<T>::value>
        struct gcd_traits_abs_defaults
        {
            inline static constexpr const T &abs(const T &val) GCD_NOEXCEPT(T) { return val; }
        };

        template <class T>
        struct gcd_traits_abs_defaults<T, false>
        {
            inline static T constexpr abs(const T &val) GCD_NOEXCEPT(T)
            {
                // This sucks, but std::abs is not constexpr :(
                return val < T(0) ? -val : val;
            }
        };

        enum class method_type
        {
            method_euclid = 0,
            method_binary = 1,
            method_mixed = 2
        };

        struct any_convert
        {
            template <class T>
            any_convert(const T &);
        };

        struct unlikely_size
        {
            char buf[9973];
        };

        unlikely_size operator <<=(any_convert, any_convert);
        unlikely_size operator >>=(any_convert, any_convert);

        template <class T>
        struct gcd_traits_defaults : public gcd_traits_abs_defaults<T>
        {
            __forceinline static constexpr unsigned make_odd(T &val) GCD_NOEXCEPT(T)
            {
                unsigned r = 0;
                while (T(0) == (val & 1u))
                {
#ifdef _MSC_VER  // VC++ can't handle operator >>= in constexpr code for some reason
                    val = val >> 1;
#else
                    val >>= 1;
#endif
                    ++r;
                }
                return r;
            }

            inline static constexpr bool less(const T &a, const T &b) GCD_NOEXCEPT(T)
            {
                return a < b;
            }

            static T &get_value();

            static const bool has_operator_left_shift_equal = sizeof(get_value() <<= 2) != sizeof(unlikely_size);
            static const bool has_operator_right_shift_equal = sizeof(get_value() >>= 2) != sizeof(unlikely_size);

            static const method_type method = std::numeric_limits<T>::is_specialized && std::numeric_limits<
                    T>::is_integer && has_operator_left_shift_equal && has_operator_right_shift_equal
                ? method_type::method_mixed
                : method_type::method_euclid;
        };

        // Default gcd_traits just inherits from defaults:
        template <class T>
        struct gcd_traits : public gcd_traits_defaults<T>
        {
        };

        // The Mixed Binary Euclid Algorithm
        // Sidi Mohamed Sedjelmaci
        // Electronic Notes in Discrete Mathematics 35 (2009) 169-176
        template <class T>
        constexpr T mixed_binary_gcd(T u, T v) GCD_NOEXCEPT(T)
        {
            if (gcd_traits<T>::less(u, v))
                constexpr_swap(u, v);

            unsigned shifts = 0;

            if (u == T(0))
                return v;
            if (v == T(0))
                return u;

            shifts = constexpr_min(gcd_traits<T>::make_odd(u), gcd_traits<T>::make_odd(v));

            while (gcd_traits<T>::less(1, v))
            {
                u %= v;
                v -= u;
                if (u == T(0))
                    return v << shifts;
                if (v == T(0))
                    return u << shifts;
                gcd_traits<T>::make_odd(u);
                gcd_traits<T>::make_odd(v);
                if (gcd_traits<T>::less(u, v))
                    constexpr_swap(u, v);
            }
            return (v == 1 ? v : u) << shifts;
        }

        /** Stein gcd (aka 'binary gcd')
         *
         * From Mathematics to Generic Programming, Alexander Stepanov, Daniel Rose
         */
        template <typename SteinDomain>
        constexpr SteinDomain Stein_gcd(SteinDomain m, SteinDomain n) GCD_NOEXCEPT(SteinDomain)
        {
            static_assert(m >= 0);
            static_assert(n >= 0);
            if (m == SteinDomain(0))
                return n;
            if (n == SteinDomain(0))
                return m;
            // m > 0 && n > 0
            unsigned d_m = gcd_traits<SteinDomain>::make_odd(m);
            unsigned d_n = gcd_traits<SteinDomain>::make_odd(n);
            // odd(m) && odd(n)
            while (m != n)
            {
                if (n > m)
                    constexpr_swap(n, m);
                m -= n;
                gcd_traits<SteinDomain>::make_odd(m);
            }
            // m == n
            m <<= constexpr_min(d_m, d_n);
            return m;
        }

        /** Euclidean algorithm
         *
         * From Mathematics to Generic Programming, Alexander Stepanov, Daniel Rose
         *
         */
        template <typename EuclideanDomain>
        inline constexpr EuclideanDomain Euclid_gcd(EuclideanDomain a, EuclideanDomain b) GCD_NOEXCEPT(EuclideanDomain)
        {
            while (b != EuclideanDomain(0))
            {
                a %= b;
                constexpr_swap(a, b);
            }
            return a;
        }

        template <typename T>
        inline constexpr typename type_traits::enable_if_c<gcd_traits<T>::method == method_type::method_mixed, T>::type
        optimal_gcd_select(T const &a, T const &b) GCD_NOEXCEPT(T)
        {
            return mixed_binary_gcd(a, b);
        }

        template <typename T>
        inline constexpr typename type_traits::enable_if_c<gcd_traits<T>::method == method_type::method_binary, T>::type
        optimal_gcd_select(T const &a, T const &b) GCD_NOEXCEPT(T)
        {
            return Stein_gcd(a, b);
        }

        template <typename T>
        inline constexpr typename type_traits::enable_if_c<gcd_traits<T>::method == method_type::method_euclid, T>::type
        optimal_gcd_select(T const &a, T const &b) GCD_NOEXCEPT(T)
        {
            return Euclid_gcd(a, b);
        }
    }

    /// Returns the greatest common divisor of |x1| and |x2|.
    template <typename Integer>
    inline constexpr Integer gcd(Integer const &a, Integer const &b) GCD_NOEXCEPT(Integer)
    {
        if (a == (std::numeric_limits<Integer>::min)())
            return a == static_cast<Integer>(0)
                ? detail::gcd_traits<Integer>::abs(b)
                : detail::gcd(static_cast<Integer>(a % b), b);
        if (b == (std::numeric_limits<Integer>::min)())
            return b == static_cast<Integer>(0)
                ? detail::gcd_traits<Integer>::abs(a)
                : detail::gcd(a, static_cast<Integer>(b % a));
        return detail::optimal_gcd_select(static_cast<Integer>(detail::gcd_traits<Integer>::abs(a)),
                                          static_cast<Integer>(detail::gcd_traits<Integer>::abs(b)));
    }

    /**
     * Knuth, The Art of Computer Programming: Volume 2, Third edition, 1998
     * Chapter 4.5.2, Algorithm C: Greatest common divisor of n integers.
     *
     * Knuth counts down from n to zero, but we naturally go from first to last.
     * We also return the termination position because it might be useful to know.
     *
     * Partly by quirk, partly by design, this algorithm is defined for n = 1,
     * because the gcd of {x} is x. It is not defined for n = 0.
     *
     * @tparam  I   Input iterator.
     * @return  The gcd of the range and the iterator position at termination.
     */
    template <typename I>
    std::pair<typename std::iterator_traits<I>::value_type, I>
    gcd_range(I first, I last) GCD_NOEXCEPT(I)
    {
        static_assert(first != last);
        typedef typename std::iterator_traits<I>::value_type T;

        T d = *first;
        ++first;
        while (d != T(1) && first != last)
        {
            d = gcd(d, *first);
            ++first;
        }
        return std::make_pair(d, first);
    }

    /// Returns the greatest common divisor of the values in the
    /// input array.
    template<typename dtype>
    dtype gcd(const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_INTEGER(dtype);
        return gcd_range(inArray.cbegin(), inArray.cend()).first;
    }
}
