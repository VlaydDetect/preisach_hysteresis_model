// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "is_range.hpp"
#include <type_traits>

namespace mc
{
    namespace hash_detail
    {

        template <class T, class E = std::true_type>
        struct has_hasher_ : std::false_type
        {
        };

        template <class T>
        struct has_hasher_<T, std::integral_constant<bool,
                                                     std::is_same<typename T::hasher, typename T::hasher>::value
                           >> : std::true_type
        {
        };

    }

    namespace hash
    {

        template <class T>
        struct is_unordered_range : std::integral_constant<
                bool, is_range<T>::value && hash_detail::has_hasher_<T>::value>
        {
        };

    }
}
