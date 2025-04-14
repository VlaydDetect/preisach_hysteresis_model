// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/StaticAssert.hpp"

namespace mc
{
    namespace utils
    {
        /// Converts the number into a string
        template <typename dtype>
        std::string num2str(dtype inNumber)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            return std::to_string(inNumber);
        }
    }
}
