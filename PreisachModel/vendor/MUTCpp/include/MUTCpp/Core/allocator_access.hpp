// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include "pointer_traits.hpp"

#define BOOST_DETAIL_ALLOC_EMPTY(T) __is_empty(T)

#if defined(_STL_DISABLE_DEPRECATED_WARNING)
_STL_DISABLE_DEPRECATED_WARNING
#endif

#pragma warning(push)
#pragma warning(disable:4996)

namespace mc
{

    template <class A>
    struct allocator_value_type
    {
        typedef typename A::value_type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_ptr
        {
            typedef typename allocator_value_type<A>::type *type;
        };

        template <class>
        struct alloc_void
        {
            typedef void type;
        };

        template <class A>
        struct alloc_ptr<A,
                         typename alloc_void<typename A::pointer>::type>
        {
            typedef typename A::pointer type;
        };

    } /* detail */

    template <class A>
    struct allocator_pointer
    {
        typedef typename detail::alloc_ptr<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_const_ptr
        {
            typedef typename pointer_traits<typename
                allocator_pointer<A>::type>::template rebind_to<const typename
                allocator_value_type<A>::type>::type type;
        };

        template <class A>
        struct alloc_const_ptr<A,
                               typename alloc_void<typename A::const_pointer>::type>
        {
            typedef typename A::const_pointer type;
        };

    } /* detail */

    template <class A>
    struct allocator_const_pointer
    {
        typedef typename detail::alloc_const_ptr<A>::type type;
    };

    namespace detail
    {

        template <class, class>
        struct alloc_to
        {
        };

        template <template<class, class...> class A, class T, class U, class... V>
        struct alloc_to<A<U, V...>, T>
        {
            typedef A<T, V...> type;
        };

        template <class A, class T, class = void>
        struct alloc_rebind
        {
            typedef typename alloc_to<A, T>::type type;
        };

        template <class A, class T>
        struct alloc_rebind<A, T,
                            typename alloc_void<typename A::template rebind<T>::other>::type>
        {
            typedef typename A::template rebind<T>::other type;
        };

    } /* detail */

    template <class A, class T>
    struct allocator_rebind
    {
        typedef typename detail::alloc_rebind<A, T>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_void_ptr
        {
            typedef typename pointer_traits<typename
                allocator_pointer<A>::type>::template
            rebind_to<void>::type type;
        };

        template <class A>
        struct alloc_void_ptr<A,
                              typename alloc_void<typename A::void_pointer>::type>
        {
            typedef typename A::void_pointer type;
        };

    } /* detail */

    template <class A>
    struct allocator_void_pointer
    {
        typedef typename detail::alloc_void_ptr<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_const_void_ptr
        {
            typedef typename pointer_traits<typename
                allocator_pointer<A>::type>::template
            rebind_to<const void>::type type;
        };

        template <class A>
        struct alloc_const_void_ptr<A,
                                    typename alloc_void<typename A::const_void_pointer>::type>
        {
            typedef typename A::const_void_pointer type;
        };

    } /* detail */

    template <class A>
    struct allocator_const_void_pointer
    {
        typedef typename detail::alloc_const_void_ptr<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_diff_type
        {
            typedef typename pointer_traits<typename
                allocator_pointer<A>::type>::difference_type type;
        };

        template <class A>
        struct alloc_diff_type<A,
                               typename alloc_void<typename A::difference_type>::type>
        {
            typedef typename A::difference_type type;
        };

    } /* detail */

    template <class A>
    struct allocator_difference_type
    {
        typedef typename detail::alloc_diff_type<A>::type type;
    };

    namespace detail
    {

#if defined(BOOST_NO_CXX11_ALLOCATOR)
template<class A, class = void>
struct alloc_size_type {
    typedef std::size_t type;
};
#else
        template <class A, class = void>
        struct alloc_size_type
        {
            typedef typename std::make_unsigned<typename
                allocator_difference_type<A>::type>::type type;
        };
#endif

        template <class A>
        struct alloc_size_type<A,
                               typename alloc_void<typename A::size_type>::type>
        {
            typedef typename A::size_type type;
        };

    } /* detail */

    template <class A>
    struct allocator_size_type
    {
        typedef typename detail::alloc_size_type<A>::type type;
    };

    namespace detail
    {

        typedef std::false_type alloc_false;

        template <class A, class = void>
        struct alloc_pocca
        {
            typedef alloc_false type;
        };

        template <class A>
        struct alloc_pocca<A,
                           typename alloc_void<typename
                               A::propagate_on_container_copy_assignment>::type>
        {
            typedef typename A::propagate_on_container_copy_assignment type;
        };

    } /* detail */

    template <class A, class = void>
    struct allocator_propagate_on_container_copy_assignment
    {
        typedef typename detail::alloc_pocca<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_pocma
        {
            typedef alloc_false type;
        };

        template <class A>
        struct alloc_pocma<A,
                           typename alloc_void<typename
                               A::propagate_on_container_move_assignment>::type>
        {
            typedef typename A::propagate_on_container_move_assignment type;
        };

    } /* detail */

    template <class A>
    struct allocator_propagate_on_container_move_assignment
    {
        typedef typename detail::alloc_pocma<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_pocs
        {
            typedef alloc_false type;
        };

        template <class A>
        struct alloc_pocs<A,
                          typename alloc_void<typename A::propagate_on_container_swap>::type>
        {
            typedef typename A::propagate_on_container_swap type;
        };

    } /* detail */

    template <class A>
    struct allocator_propagate_on_container_swap
    {
        typedef typename detail::alloc_pocs<A>::type type;
    };

    namespace detail
    {

        template <class A, class = void>
        struct alloc_equal
        {
            typedef typename std::is_empty<A>::type type;
        };

        template <class A>
        struct alloc_equal<A,
                           typename alloc_void<typename A::is_always_equal>::type>
        {
            typedef typename A::is_always_equal type;
        };

    } /* detail */

    template <class A>
    struct allocator_is_always_equal
    {
        typedef typename detail::alloc_equal<A>::type type;
    };

    template <class A>
    inline typename allocator_pointer<A>::type
    allocator_allocate(A &a, typename allocator_size_type<A>::type n)
    {
        return a.allocate(n);
    }

    template <class A>
    inline void
    allocator_deallocate(A &a, typename allocator_pointer<A>::type p,
                         typename allocator_size_type<A>::type n)
    {
        a.deallocate(p, n);
    }

    namespace detail
    {

        template <class>
        struct alloc_no
        {
            char x, y;
        };

        template <class A>
        class alloc_has_allocate
        {
            template <class O>
            static auto check(int)
                -> alloc_no<decltype(std::declval<O &>().allocate(std::declval<typename
                                                                      allocator_size_type<A>::type>(),
                                                                  std::declval<typename
                                                                      allocator_const_void_pointer<A>::type>()))>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<A>(0)) > 1;
        };

    } /* detail */

    template <class A>
    inline typename std::enable_if<detail::alloc_has_allocate<A>::value,
                                   typename allocator_pointer<A>::type>::type
    allocator_allocate(A &a, typename allocator_size_type<A>::type n,
                       typename allocator_const_void_pointer<A>::type h)
    {
        return a.allocate(n, h);
    }

    template <class A>
    inline typename std::enable_if<!detail::alloc_has_allocate<A>::value,
                                   typename allocator_pointer<A>::type>::type
    allocator_allocate(A &a, typename allocator_size_type<A>::type n,
                       typename allocator_const_void_pointer<A>::type)
    {
        return a.allocate(n);
    }

    namespace detail
    {

        template <class A, class T, class... Args>
        class alloc_has_construct
        {
            template <class O>
            static auto check(int)
                -> alloc_no<decltype(std::declval<O &>().construct(std::declval<T *>(),
                                                                   std::declval<Args &&>()...))>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<A>(0)) > 1;
        };

        template <bool, class = void>
        struct alloc_if
        {
        };

        template <class T>
        struct alloc_if<true, T>
        {
            typedef T type;
        };

    } /* detail */

    template <class A, class T, class... Args>
    inline std::enable_if_t<detail::alloc_has_construct<A, T,
                                                        Args...>::value>
    allocator_construct(A &a, T *p, Args &&... args)
    {
        a.construct(p, std::forward<Args>(args)...);
    }

    template <class A, class T, class... Args>
    inline std::enable_if_t<!detail::alloc_has_construct<A, T,
                                                         Args...>::value>
    allocator_construct(A &, T *p, Args &&... args)
    {
        ::new((void *)p) T(std::forward<Args>(args)...);
    }

    namespace detail
    {

        template <class A, class T>
        class alloc_has_destroy
        {
            template <class O>
            static auto check(int)
                -> alloc_no<decltype(std::declval<O &>().destroy(std::declval<T *>()))>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<A>(0)) > 1;
        };

    } /* detail */

    template <class A, class T>
    inline typename detail::alloc_if<detail::alloc_has_destroy<A, T>::value>::type
    allocator_destroy(A &a, T *p)
    {
        a.destroy(p);
    }

    template <class A, class T>
    inline typename detail::alloc_if<!detail::alloc_has_destroy<A, T>::value>::type
    allocator_destroy(A &, T *p)
    {
        p->~T();
        (void)p;
    }

    namespace detail
    {

        template <class A>
        class alloc_has_max_size
        {
            template <class O>
            static auto check(int)
                -> alloc_no<decltype(std::declval<const O &>().max_size())>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<A>(0)) > 1;
        };

    } /* detail */

    template <class A>
    inline typename detail::alloc_if<detail::alloc_has_max_size<A>::value,
                                     typename allocator_size_type<A>::type>::type
    allocator_max_size(const A &a) noexcept
    {
        return a.max_size();
    }

    template <class A>
    inline typename detail::alloc_if<!detail::alloc_has_max_size<A>::value,
                                     typename allocator_size_type<A>::type>::type
    allocator_max_size(const A &) noexcept
    {
        return (std::numeric_limits<typename
                allocator_size_type<A>::type>::max)() /
            sizeof(typename allocator_value_type<A>::type);
    }

    namespace detail
    {

        template <class A>
        class alloc_has_soccc
        {
            template <class O>
            static auto check(int) -> alloc_no<decltype(std::declval<const
                O &>().select_on_container_copy_construction())>;

            template <class>
            static char check(long);

        public:
            static constexpr bool value = sizeof(check<A>(0)) > 1;
        };

    } /* detail */

    template <class A>
    inline typename detail::alloc_if<detail::alloc_has_soccc<A>::value, A>::type
    allocator_select_on_container_copy_construction(const A &a)
    {
        return a.select_on_container_copy_construction();
    }

    template <class A>
    inline typename detail::alloc_if<!detail::alloc_has_soccc<A>::value, A>::type
    allocator_select_on_container_copy_construction(const A &a)
    {
        return a;
    }

    template <class A, class T>
    inline void
    allocator_destroy_n(A &a, T *p, std::size_t n)
    {
        while (n > 0)
        {
            allocator_destroy(a, p + --n);
        }
    }

    namespace detail
    {

        template <class A, class T>
        class alloc_destroyer
        {
        public:
            alloc_destroyer(A &a, T *p) noexcept :
                a_(a), p_(p), n_(0)
            {
            }

            ~alloc_destroyer()
            {
                allocator_destroy_n(a_, p_, n_);
            }

            std::size_t &size() noexcept
            {
                return n_;
            }

        private:
            alloc_destroyer(const alloc_destroyer &);
            alloc_destroyer &operator=(const alloc_destroyer &);

            A &a_;
            T *p_;
            std::size_t n_;
        };

    } /* detail */

    template <class A, class T>
    inline void
    allocator_construct_n(A &a, T *p, std::size_t n)
    {
        detail::alloc_destroyer<A, T> d(a, p);
        for (std::size_t &i = d.size(); i < n; ++i)
        {
            allocator_construct(a, p + i);
        }
        d.size() = 0;
    }

    template <class A, class T>
    inline void
    allocator_construct_n(A &a, T *p, std::size_t n, const T *l, std::size_t m)
    {
        detail::alloc_destroyer<A, T> d(a, p);
        for (std::size_t &i = d.size(); i < n; ++i)
        {
            allocator_construct(a, p + i, l[i % m]);
        }
        d.size() = 0;
    }

    template <class A, class T, class I>
    inline void
    allocator_construct_n(A &a, T *p, std::size_t n, I b)
    {
        detail::alloc_destroyer<A, T> d(a, p);
        for (std::size_t &i = d.size(); i < n; void(++i), void(++b))
        {
            allocator_construct(a, p + i, *b);
        }
        d.size() = 0;
    }

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
    template <class A>
    using allocator_value_type_t = typename allocator_value_type<A>::type;

    template <class A>
    using allocator_pointer_t = typename allocator_pointer<A>::type;

    template <class A>
    using allocator_const_pointer_t = typename allocator_const_pointer<A>::type;

    template <class A>
    using allocator_void_pointer_t = typename allocator_void_pointer<A>::type;

    template <class A>
    using allocator_const_void_pointer_t =
    typename allocator_const_void_pointer<A>::type;

    template <class A>
    using allocator_difference_type_t =
    typename allocator_difference_type<A>::type;

    template <class A>
    using allocator_size_type_t = typename allocator_size_type<A>::type;

    template <class A>
    using allocator_propagate_on_container_copy_assignment_t =
    typename allocator_propagate_on_container_copy_assignment<A>::type;

    template <class A>
    using allocator_propagate_on_container_move_assignment_t =
    typename allocator_propagate_on_container_move_assignment<A>::type;

    template <class A>
    using allocator_propagate_on_container_swap_t =
    typename allocator_propagate_on_container_swap<A>::type;

    template <class A>
    using allocator_is_always_equal_t =
    typename allocator_is_always_equal<A>::type;

    template <class A, class T>
    using allocator_rebind_t = typename allocator_rebind<A, T>::type;
#endif

} /* boost */

#pragma warning(pop)
#if defined(_STL_RESTORE_DEPRECATED_WARNING)
_STL_RESTORE_DEPRECATED_WARNING
#endif
