// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <bitset>
#include <string>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"

namespace mc
{
    /// Return the binary representation of the input number as a string.
    template <typename dtype>
    std::string binaryRepr(dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::bitset<DtypeInfo<dtype>::bits()>(inValue).to_string();
    }
}
