// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <vector>

namespace mc
{
    template <typename T, std::input_iterator Iterator>
    Eigen::RowVectorX<T> fromiter(Iterator first, Iterator last)
    {
        const std::vector<T> values(first, last);
        Eigen::RowVectorX<T> result(static_cast<Eigen::Index>(values.size()));
        std::copy(values.begin(), values.end(), result.begin());
        return result;
    }
}
