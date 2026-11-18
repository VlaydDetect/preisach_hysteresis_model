// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/fromiter.hpp"
#include <sstream>
#include <string>

namespace mc
{
    template <typename T>
    Eigen::RowVectorX<T> fromstring(const std::string &text, char separator = ' ')
    {
        std::istringstream input(text);
        std::vector<T> values;
        for (std::string segment; std::getline(input, segment, separator);)
        {
            std::istringstream elements(segment);
            for (T value{}; elements >> value;) values.push_back(value);
        }
        return fromiter<T>(values.begin(), values.end());
    }
}
