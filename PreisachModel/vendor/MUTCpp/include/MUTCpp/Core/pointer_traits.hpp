// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cstddef>

namespace mc
{
    namespace detail
    {

        struct ptr_none
        {
        };

        template <class>
        struct ptr_valid
        {
            typedef void type;
        };

        template <class>
        struct ptr_first
        {
            typedef ptr_none type;
        };

        template <template<class, class...> class T, class U, class... Args>
        struct ptr_first<T<U, Args...>>
        {
            typedef U type;
        };

        template <class T, class = void>
        struct ptr_element
        {
            typedef typename ptr_first<T>::type type;
        };

        template <class T>
        struct ptr_element<T, typename ptr_valid<typename T::element_type>::type>
        {
            typedef typename T::element_type type;
        };

        template <class, class = void>
        struct ptr_difference
        {
            typedef std::ptrdiff_t type;
        };

        template <class T>
        struct ptr_difference<T,
                              typename ptr_valid<typename T::difference_type>::type>
        {
            typedef typename T::difference_type type;
        };

        template <class, class>
        struct ptr_transform
        {
        };

        template <template<class, class...> class T, class U, class... Args, class V>
        struct ptr_transform<T<U, Args...>, V>
        {
            typedef T<V, Args...> type;
        };

        template <class T, class U, class = void>
        struct ptr_rebind
            : ptr_transform<T, U>
        {
        };

        template <class T, class U>
        struct ptr_rebind<T, U,
                          typename ptr_valid<typename T::template rebind<U>>::type>
        {
            typedef typename T::template rebind<U> type;
        };

#if !defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <class T, class E>
        class ptr_to_expr
        {
            template <class>
            struct result
            {
                char x, y;
            };

            static E &source();

            template <class O>
            static auto check(int) -> result<decltype(O::pointer_to(source()))>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<T>(0)) > 1;
        };

        template <class T, class E>
        struct ptr_to_expr<T *, E>
        {
            static constexpr bool value = true;
        };

        template <class T, class E>
        struct ptr_has_to
        {
            static constexpr bool value = ptr_to_expr<T, E>::value;
        };
#else
template<class, class>
struct ptr_has_to {
    static constexpr bool value = true;
};
#endif

        template <class T>
        struct ptr_has_to<T, void>
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct ptr_has_to<T, const void>
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct ptr_has_to<T, volatile void>
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct ptr_has_to<T, const volatile void>
        {
            static constexpr bool value = false;
        };

        template <class T, class E, bool = ptr_has_to<T, E>::value>
        struct ptr_to
        {
        };

        template <class T, class E>
        struct ptr_to<T, E, true>
        {
            static T pointer_to(E &v)
            {
                return T::pointer_to(v);
            }
        };

        template <class T>
        struct ptr_to<T *, T, true>
        {
            static T *pointer_to(T &v) noexcept
            {
                return std::addressof(v);
            }
        };

        template <class T, class E>
        struct ptr_traits
            : ptr_to<T, E>
        {
            typedef T pointer;
            typedef E element_type;
            typedef typename ptr_difference<T>::type difference_type;

            template <class U>
            struct rebind_to
                : ptr_rebind<T, U>
            {
            };

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
            template <class U>
            using rebind = typename rebind_to<U>::type;
#endif
        };

        template <class T>
        struct ptr_traits<T, ptr_none>
        {
        };

    } /* detail */

    template <class T>
    struct pointer_traits
        : detail::ptr_traits<T, typename detail::ptr_element<T>::type>
    {
    };

    template <class T>
    struct pointer_traits<T *>
        : detail::ptr_to<T *, T>
    {
        typedef T *pointer;
        typedef T element_type;
        typedef std::ptrdiff_t difference_type;

        template <class U>
        struct rebind_to
        {
            typedef U *type;
        };

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
        template <class U>
        using rebind = typename rebind_to<U>::type;
#endif
    };

    template <class T>
    constexpr inline T *
    to_address(T *v) noexcept
    {
        return v;
    }

    template <class T>
    inline typename pointer_traits<T>::element_type *
    to_address(const T &v) noexcept
    {
        return to_address(v.operator->());
    }

}
