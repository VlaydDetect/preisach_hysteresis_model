// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <type_traits>

#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    // all_equals
    template<typename T>
    bool all_equals(T a, T b)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            return utils::essentiallyEqual<T>(a, b);
        }
        
        return a == b;
    }

    template<typename T, typename... Args>
    bool all_equals(T a, T b, Args... args)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            if (!utils::essentiallyEqual(a, b)) return false;
        }
        else
        {
            if (a != b) return false;
        }

        return all_equals(b, args...);
    }

    // all_less
    template<typename T>
    bool all_less(T a, T b)
    {
        return a < b;
    }

    template<typename T, typename... Args>
    bool all_less(T a, T b, Args... args)
    {
        return a < b && all_less(b, args...);
    }

    // all_greater
    template<typename T>
    bool all_greater(T a, T b)
    {
        return a > b;
    }

    template<typename T, typename... Args>
    bool all_greater(T a, T b, Args... args)
    {
        return a > b && all_less(b, args...);
    }
}
