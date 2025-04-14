// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "is_range.hpp"
#include <type_traits>

#include <iterator>

namespace mc
{
    namespace hash_detail
    {

        template <class It, class T, class S>
        std::integral_constant<bool, std::is_same<typename std::iterator_traits<It>::value_type, T>::value &&
                               std::is_integral<S>::value>
        is_contiguous_range_check(It first, It last, T const *, T const *, S);

        template <class T>
        decltype( is_contiguous_range_check(std::declval<T const &>().begin(), std::declval<T const &>().end(),
                                            std::declval<T const &>().data(),
                                            std::declval<T const &>().data() + std::declval<T const &>().size(),
                                            std::declval<T const &>().size()) ) is_contiguous_range_(int);
        template <class T>
        std::false_type is_contiguous_range_(...);

        template <class T>
        struct is_contiguous_range : decltype( hash_detail::is_contiguous_range_<T>(0) )
        {
        };

    }

    namespace hash
    {

        template <class T>
        struct is_contiguous_range : std::integral_constant<
                bool, is_range<T>::value && hash_detail::is_contiguous_range<T>::value>
        {
        };

    }
}
