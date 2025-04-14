// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <locale>

namespace mc { namespace detail {
    template <typename T> T make_non_const(T t) { return t; }
}}

#define BOOST_DYNAMIC_BITSET_CTYPE_FACET(ch, name, loc) \
            const std::ctype<ch> & name =               \
            std::use_facet<std::ctype<ch>>(loc)         /**/

#define BOOST_DYNAMIC_BITSET_WIDEN_CHAR(fac, c) (fac.widen(c))
