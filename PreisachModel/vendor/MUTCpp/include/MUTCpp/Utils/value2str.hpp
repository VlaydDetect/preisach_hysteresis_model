// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>
#include <sstream>
#include <string>

#include "Core/StaticAssert.hpp"

namespace mc
{
    namespace utils
    {
        /// Converts the value into a string
        template<typename dtype>
        std::string value2str(dtype inValue)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::stringstream ss;
            ss << inValue;
            return ss.str();
        }
    }
}
