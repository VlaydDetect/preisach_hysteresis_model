// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <sstream>
#include <string_view>

#include "Matrix.hpp"

namespace mc
{
    /// Construct an array from data in a string
    /// @param inStr
    /// @param inSep: Delimiter separator between values in the string
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> fromstring(const std::string &inStr, const char inSep = ' ')
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::istringstream inputStream(inStr);
        auto values = std::vector<dtype>{};
        dtype value{};
        for (std::string segment; std::getline(inputStream, segment, inSep);)
        {
            if (!inputStream.fail())
            {
                std::istringstream segmentStream(segment);
                while (segmentStream >> value)
                {
                    if (!inputStream.fail())
                    {
                        values.push_back(value);
                    }
                }
            }
        }

        return Matrix<dtype>(values);
    }
}
