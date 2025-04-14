// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "Functions/unique.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the indices of the bins to which each value in input array belongs.
    /// @param x: Input array to be binned.
    /// @param bins: Array of bins.
    /// @return Matrix
    template <typename dtype1, typename dtype2>
    Matrix<uint32> digitize(const Matrix<dtype1> &x, const Matrix<dtype2> &bins)
    {
        const auto uniqueBins = unique(bins);
        auto result = Matrix<uint32>(1, x.size());
        result.fill(0);

        typename decltype(result)::size_type idx{0};
        std::for_each(x.begin(),
                      x.end(),
                      [&uniqueBins, &result, &idx](const auto value)
                      {
                          const auto upperBin = std::upper_bound(uniqueBins.begin(), uniqueBins.end(), value);
                          result[idx++] = static_cast<uint32>(std::distance(uniqueBins.begin(), upperBin));
                      });

        return result;
    }
}
