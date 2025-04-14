// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cstddef>

namespace mc
{

    namespace hash
    {
        template <class T>
        struct is_range;
        template <class T>
        struct is_contiguous_range;
        template <class T>
        struct is_unordered_range;
        template <class T>
        struct is_described_class;
        template <class T>
        struct is_tuple_like;
    }

    template <class T>
    struct Hash;

    template <class T>
    void hash_combine(std::size_t &seed, T const &v);

    template <class It>
    void hash_range(std::size_t &, It, It);
    template <class It>
    std::size_t hash_range(It, It);

    template <class It>
    void hash_unordered_range(std::size_t &, It, It);
    template <class It>
    std::size_t hash_unordered_range(It, It);

}
