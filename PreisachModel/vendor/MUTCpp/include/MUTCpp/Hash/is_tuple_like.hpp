// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <type_traits>
#include <utility>

namespace mc
{
    namespace hash_detail
    {

        template <class T, class E = std::true_type>
        struct is_tuple_like_ : std::false_type
        {
        };

        template <class T>
        struct is_tuple_like_<T, std::integral_constant<
                                  bool, std::tuple_size<T>::value == std::tuple_size<T>::value>> : std::true_type
        {
        };

    }

    namespace hash
    {

        template <class T>
        struct is_tuple_like : hash_detail::is_tuple_like_<T>
        {
        };

    }
}
