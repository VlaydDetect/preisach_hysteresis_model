// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>

namespace mc
{
    namespace hash_detail
    {
        template <class T, class It>
        std::integral_constant<bool, !std::is_same<std::remove_cv_t<T>, typename std::iterator_traits<It>::value_type>::value>
        is_range_check(It first, It last);

        template <class T>
        decltype( is_range_check<T>(std::declval<T const &>().begin(), std::declval<T const &>().end()) )
        is_range_(int);
        template <class T>
        std::false_type is_range_(...);

    } // namespace hash_detail

    namespace hash
    {
        template <class T>
        struct is_range : decltype( hash_detail::is_range_<T>(0) )
        {
        };
    }
}
