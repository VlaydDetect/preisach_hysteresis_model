// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.
#pragma once

#include <utility>

#include "Hash/hash_fwd.hpp"
#include "Hash/is_tuple_like.hpp"
#include "Hash/is_range.hpp"

namespace mc
{
    namespace hash_detail
    {

        template <std::size_t I, typename T>
        inline
        std::enable_if_t<I == std::tuple_size_v<T>>
        hash_combine_tuple_like(std::size_t &, T const &)
        {
        }

        template <std::size_t I, typename T>
        inline
        std::enable_if_t<I < std::tuple_size_v<T>>
        hash_combine_tuple_like(std::size_t &seed, T const &v)
        {
            using std::get;
            hash_combine(seed, get<I>(v));

            hash_detail::hash_combine_tuple_like<I + 1>(seed, v);
        }

        template <typename T>
        inline std::size_t hash_tuple_like(T const &v)
        {
            std::size_t seed = 0;

            hash_detail::hash_combine_tuple_like<0>(seed, v);

            return seed;
        }

    } // namespace hash_detail

    template <class T>
    inline
    std::enable_if_t<
        mc::hash::is_tuple_like<T>::value && !mc::hash::is_range<T>::value,
        std::size_t>
    hash_value(T const &v)
    {
        return hash_detail::hash_tuple_like(v);
    }

}
