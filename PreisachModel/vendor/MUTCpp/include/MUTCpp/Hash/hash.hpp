// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.
#pragma once

#include <string>
#include <iterator>
#include <complex>
#include <memory>
#include <utility>
#include <limits>
#include <climits>
#include <cstring>
#include <typeindex>
#include <system_error>
#include <optional>
#include <variant>
#include <string_view>

#include "hash_fwd.hpp"
#include "is_range.hpp"
#include "is_contiguous_range.hpp"
#include "is_unordered_range.hpp"
#include "detail/hash_integral.hpp"
#include "detail/hash_tuple_like.hpp"
#include "detail/hash_mix.hpp"
#include "detail/hash_range.hpp"
#include <type_traits>

namespace mc
{

    //
    // mc::hash_value
    //

    // integral types
    //   in detail/hash_integral.hpp

    // enumeration types

    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, std::size_t>
    hash_value(T v)
    {
        // This should in principle return the equivalent of
        //
        // mc::hash_value( to_underlying(v) );
        //
        // However, the C++03 implementation of underlying_type,
        //
        // conditional<is_signed<T>, make_signed<T>, make_unsigned<T>>::type::type
        //
        // generates a legitimate -Wconversion warning in is_signed,
        // because -1 is not a valid enum value when all the enumerators
        // are nonnegative.
        //
        // So the legacy implementation will have to do for now.

        return static_cast<std::size_t>(v);
    }

    // floating point types

    namespace hash_detail
    {
        template <class T,
                  std::size_t Bits = sizeof(T) * CHAR_BIT,
                  int Digits = std::numeric_limits<T>::digits>
        struct hash_float_impl;

        // float
        template <class T, int Digits>
        struct hash_float_impl<T, 32, Digits>
        {
            static std::size_t fn(T v)
            {
                std::uint32_t w;
                std::memcpy(&w, &v, sizeof(v));

                return w;
            }
        };

        // double
        template <class T, int Digits>
        struct hash_float_impl<T, 64, Digits>
        {
            static std::size_t fn(T v)
            {
                std::uint64_t w;
                std::memcpy(&w, &v, sizeof(v));

                return hash_value(w);
            }
        };

        // 80 bit long double in 12 bytes
        template <class T>
        struct hash_float_impl<T, 96, 64>
        {
            static std::size_t fn(T v)
            {
                std::uint64_t w[2] = {};
                std::memcpy(&w, &v, 80 / CHAR_BIT);

                std::size_t seed = 0;

                seed = hash_value(w[0]) + hash_detail::hash_mix(seed);
                seed = hash_value(w[1]) + hash_detail::hash_mix(seed);

                return seed;
            }
        };

        // 80 bit long double in 16 bytes
        template <class T>
        struct hash_float_impl<T, 128, 64>
        {
            static std::size_t fn(T v)
            {
                std::uint64_t w[2] = {};
                std::memcpy(&w, &v, 80 / CHAR_BIT);

                std::size_t seed = 0;

                seed = hash_value(w[0]) + hash_detail::hash_mix(seed);
                seed = hash_value(w[1]) + hash_detail::hash_mix(seed);

                return seed;
            }
        };

        // 128 bit long double
        template <class T, int Digits>
        struct hash_float_impl<T, 128, Digits>
        {
            static std::size_t fn(T v)
            {
                std::uint64_t w[2];
                std::memcpy(&w, &v, sizeof(v));

                std::size_t seed = 0;

                seed = hash_value(w[0]) + hash_detail::hash_mix(seed);
                seed = hash_value(w[1]) + hash_detail::hash_mix(seed);

                return seed;
            }
        };

    } // namespace hash_detail

    template <typename T>
    std::enable_if_t<std::is_floating_point<T>::value, std::size_t>
    hash_value(T v)
    {
        return mc::hash_detail::hash_float_impl<T>::fn(v + 0);
    }

    // pointer types

    // `x + (x >> 3)` adjustment by Alberto Barbati and Dave Harris.
    template <class T>
    std::size_t hash_value(T *const&v)
    {
        std::uintptr_t x = reinterpret_cast<std::uintptr_t>(v);
        return mc::hash_value(x + (x >> 3));
    }

    // array types

    template <class T, std::size_t N>
    inline std::size_t hash_value(T const (&x)[N])
    {
        return mc::hash_range(x, x + N);
    }

    template <class T, std::size_t N>
    inline std::size_t hash_value(T (&x)[N])
    {
        return mc::hash_range(x, x + N);
    }

    // complex

    template <class T>
    std::size_t hash_value(std::complex<T> const &v)
    {
        const std::size_t re = mc::Hash<T>()(v.real());
        const std::size_t im = mc::Hash<T>()(v.imag());

        return re + hash_detail::hash_mix(im);
    }

    // pair

    template <class A, class B>
    std::size_t hash_value(std::pair<A, B> const &v)
    {
        std::size_t seed = 0;

        mc::hash_combine(seed, v.first);
        mc::hash_combine(seed, v.second);

        return seed;
    }

    // ranges (list, set, deque...)

    template <typename T>
    std::enable_if_t<hash::is_range<T>::value && !hash::is_contiguous_range<T>::value && !hash::is_unordered_range<
                         T>::value, std::size_t>
    hash_value(T const &v)
    {
        return mc::hash_range(v.begin(), v.end());
    }

    // contiguous ranges (string, vector, array)

    template <typename T>
    std::enable_if_t<hash::is_contiguous_range<T>::value, std::size_t>
    hash_value(T const &v)
    {
        return mc::hash_range(v.data(), v.data() + v.size());
    }

    // unordered ranges (unordered_set, unordered_map)

    template <typename T>
    std::enable_if_t<hash::is_unordered_range<T>::value, std::size_t>
    hash_value(T const &v)
    {
        return mc::hash_unordered_range(v.begin(), v.end());
    }

    // resolve ambiguity with unconstrained stdext::hash_value in <xhash> :-/

    template <template<class...> class L, class... T>
    std::enable_if_t<hash::is_range<L<T...>>::value && !hash::is_contiguous_range<L<T...>>::value && !
                     hash::is_unordered_range<L<T...>>::value, std::size_t>
    hash_value(L<T...> const &v)
    {
        return mc::hash_range(v.begin(), v.end());
    }

    // contiguous ranges (string, vector, array)

    template <template<class...> class L, class... T>
    std::enable_if_t<hash::is_contiguous_range<L<T...>>::value, std::size_t>
    hash_value(L<T...> const &v)
    {
        return mc::hash_range(v.data(), v.data() + v.size());
    }

    template <template<class, std::size_t> class L, class T, std::size_t N>
    std::enable_if_t<hash::is_contiguous_range<L<T, N>>::value, std::size_t>
    hash_value(L<T, N> const &v)
    {
        return mc::hash_range(v.data(), v.data() + v.size());
    }

    // unordered ranges (unordered_set, unordered_map)

    template <template<class...> class L, class... T>
    std::enable_if_t<hash::is_unordered_range<L<T...>>::value, std::size_t>
    hash_value(L<T...> const &v)
    {
        return mc::hash_unordered_range(v.begin(), v.end());
    }

    // std::unique_ptr, std::shared_ptr

    template <typename T>
    std::size_t hash_value(std::shared_ptr<T> const &x)
    {
        return mc::hash_value(x.get());
    }

    template <typename T, typename Deleter>
    std::size_t hash_value(std::unique_ptr<T, Deleter> const &x)
    {
        return mc::hash_value(x.get());
    }

    // std::type_index

    inline std::size_t hash_value(std::type_index const &v)
    {
        return v.hash_code();
    }

    // std::error_code, std::error_condition

    inline std::size_t hash_value(std::error_code const &v)
    {
        std::size_t seed = 0;

        mc::hash_combine(seed, v.value());
        mc::hash_combine(seed, &v.category());

        return seed;
    }

    inline std::size_t hash_value(std::error_condition const &v)
    {
        std::size_t seed = 0;

        mc::hash_combine(seed, v.value());
        mc::hash_combine(seed, &v.category());

        return seed;
    }

    // std::nullptr_t

    template <typename T>
    std::enable_if_t<std::is_same<T, std::nullptr_t>::value, std::size_t>
    hash_value(T const & /*v*/)
    {
        return mc::hash_value(static_cast<void *>(nullptr));
    }

    // std::optional

    template <typename T>
    std::size_t hash_value(std::optional<T> const &v)
    {
        if (!v)
        {
            // Arbitrary value for empty optional.
            return 0x12345678;
        }
        return mc::Hash<T>()(*v);
    }

    // std::variant

    inline std::size_t hash_value(std::monostate)
    {
        return 0x87654321;
    }

    template <typename... Types>
    std::size_t hash_value(std::variant<Types...> const &v)
    {
        std::size_t seed = 0;

        hash_combine(seed, v.index());
        std::visit([&seed](auto &&x) { hash_combine(seed, x); }, v);

        return seed;
    }

    //
    // mc::hash_combine
    //

    template <class T>
    inline void hash_combine(std::size_t &seed, T const &v)
    {
        seed = hash_detail::hash_mix(seed + 0x9e3779b9 + mc::Hash<T>()(v));
    }

    //
    // mc::hash_range
    //

    template <class It>
    inline void hash_range(std::size_t &seed, It first, It last)
    {
        seed = hash_detail::hash_range(seed, first, last);
    }

    template <class It>
    inline std::size_t hash_range(It first, It last)
    {
        std::size_t seed = 0;

        hash_range(seed, first, last);

        return seed;
    }

    //
    // mc::hash_unordered_range
    //

    template <class It>
    inline void hash_unordered_range(std::size_t &seed, It first, It last)
    {
        std::size_t r = 0;
        std::size_t const s2(seed);

        for (; first != last; ++first)
        {
            std::size_t s3(s2);

            hash_combine<typename std::iterator_traits<It>::value_type>(s3, *first);

            r += s3;
        }

        seed += r;
    }

    template <class It>
    inline std::size_t hash_unordered_range(It first, It last)
    {
        std::size_t seed = 0;

        hash_unordered_range(seed, first, last);

        return seed;
    }

    //
    // mc::hash
    //

    template <class T>
    struct Hash
    {
        typedef T argument_type;
        typedef std::size_t result_type;

        std::size_t operator()(T const &val) const
        {
            return hash_value(val);
        }
    };

    // Dinkumware has stdext::hash_value for basic_string in <xhash> :-/

    template <class E, class T, class A>
    struct Hash<std::basic_string<E, T, A>>
    {
        typedef std::basic_string<E, T, A> argument_type;
        typedef std::size_t result_type;

        std::size_t operator()(std::basic_string<E, T, A> const &val) const
        {
            return mc::hash_value(val);
        }
    };

    // mc::unordered::hash_is_avalanching

    namespace unordered
    {
        template <class T>
        struct hash_is_avalanching;

        template <class Ch>
        struct hash_is_avalanching<mc::Hash<std::basic_string<Ch>>> : std::is_integral<Ch>
        {
        };

        template <class Ch>
        struct hash_is_avalanching<mc::Hash<std::basic_string_view<Ch>>> : std::is_integral<Ch>
        {
        };
    }
}
